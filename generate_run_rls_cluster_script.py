# Generate runnable scripts for benchmarking
import argparse
import yaml
parser = argparse.ArgumentParser("Generation script", add_help=False)
parser.add_argument("-k","--extra_kmp",action="store_true",default=False,help="llm extra kmp configuration")
parser.add_argument("-d","--deepspeed",action="store_true",default=False,help="only for deepspeed")
parser.add_argument("--acc_cluster1",action="store_true",default=False,help="only for rls track")
parser.add_argument("--acc_cluster2",action="store_true",default=False,help="only for rls track")
parser.add_argument("--acc_cluster3",action="store_true",default=False,help="only for rls track")
parser.add_argument("--acc_cluster4",action="store_true",default=False,help="only for rls track")
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
        lines.append("export log_dir=/mnt/aitrgdata/mint/ipex23prepare")
        lines.append("export HF_HOME=/mnt/aitrgdata/datasets/huggingface")
        lines.append("export TRANSFORMERS_OFFLINE=0")
        lines.append("bash token.sh")

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
                                                            --benchmark -m {model_id} --ipex-smooth-quant  --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency --profile  \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:  
                                                if 'falcon' in model_id:
                                                    lines.append(f"OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  \
                                                                --benchmark -m {model_id} --ipex-smooth-quant --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency --profile --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                    2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log") 
                                                elif 'mpt' in model_id:
                                                    lines.append(f"python run.py  \
                                                                --benchmark -m {model_id} --ipex-smooth-quant --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency --profile  --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                      
                                                
                                                else: 
                                                    lines.append(f"python run.py  \
                                                                --benchmark -m {model_id} --ipex-smooth-quant --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency --profile   \
                                                                    2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")       
                                        elif 'bf16' in dtype:
                                            if beam == True:   
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-smooth-quant --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --batch-size {bs} --quant-with-amp --token-latency --profile    \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-smooth-quant --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --quant-with-amp --token-latency --profile    \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                           
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
 
   
        if mode.endswith('autotune1'):
            lines.append("# Run Workload") 
            lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/meta-llama/Llama-2-7b-hf/")
            lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/meta-llama/Llama-2-13b-hf/")
            lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/facebook/opt-30b/")
            # lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/meta-llama/Llama-2-13b-hf/")
            # lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/EleutherAI/gpt-j-6b/")
            # lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/facebook/opt-1.3b/")
            lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/THUDM/chatglm3-6b/")

            lines.append(f"python run.py -m meta-llama/Llama-2-7b-hf --ipex-smooth-quant --batch-size 56 --calib-len 2048 --fallback-add --alpha auto --init-alpha 0.8 --alpha-min 0.8 --alpha-max 0.99 --alpha-step 0.01 --shared-criterion 'mean' --output-dir {data['modelargs'][mode]['outputdir']}/meta-llama/Llama-2-7b-hf/  \
                         2>&1 | tee -a $log_dir/llm_default_meta-llama-Llama-2-7b-hf_static8_autotune_SPR.log")
            
            lines.append("wait")

            lines.append(f"python run.py -m meta-llama/Llama-2-13b-hf --ipex-smooth-quant --batch-size 56 --calib-len 1024 --fallback-add --calib-shuffle --calib-padding --alpha auto --init-alpha 0.8 --alpha-min 0.75 --alpha-max 0.99 --alpha-step 0.01 --output-dir {data['modelargs'][mode]['outputdir']}/meta-llama/Llama-2-13b-hf/  \
                         2>&1 | tee -a $log_dir/llm_default_meta-llama-Llama-2-13b-hf_static8_autotune_SPR.log")
            lines.append("wait")
            # lines.append("python run.py -m EleutherAI/gpt-j-6b --ipex-smooth-quant --output-dir {data['modelargs'][mode]['outputdir']}/EleutherAI/gpt-j-6b/ --batch-size 56 --calib-iters 100 --calib-shuffle --fallback-add --alpha 0.85 \
            #              2>&1 | tee -a $log_dir/llm_default_EleutherAI-gpt-j-6b_static8_autotune_SPR.log")

            lines.append(f"python run.py -m facebook/opt-30b --ipex-smooth-quant --batch-size 56 --calib-iters 100 --calib-shuffle --output-dir {data['modelargs'][mode]['outputdir']}/facebook/opt-30b/  \
                         2>&1 | tee -a $log_dir/llm_default_facebook-opt-30b_static8_autotune_SPR.log")
            lines.append("wait")

            lines.append(f"python run.py -m THUDM/chatglm3-6b --ipex-smooth-quant --batch-size 56 --calib-iters 100 --calib-shuffle --alpha 0.85 --output-dir {data['modelargs'][mode]['outputdir']}/THUDM/chatglm3-6b/  \
                         2>&1 | tee -a $log_dir/llm_default_THUDM-chatglm3-6b_static8_autotune_SPR.log")
                        
            lines.append("wait")

        if mode.endswith('autotune2'):
            lines.append("# Run Workload")
            # lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/meta-llama/Llama-2-70b-hf/") 
            # lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/facebook/opt-1.3b/")
            lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/EleutherAI/gpt-j-6b/")

            # lines.append(f"python run.py -m meta-llama/Llama-2-70b-hf --ipex-smooth-quant --batch-size 56 --calib-shuffle --fallback-add --alpha 0.8 --output-dir {data['modelargs'][mode]['outputdir']}/meta-llama/Llama-2-70b-hf/  \
            #              2>&1 | tee -a $log_dir/llm_default_meta-llama-Llama-2-70b-hf_static8_autotune_SPR.log")
            # lines.append("wait")            
            # lines.append(f"python run.py -m facebook/opt-1.3b --ipex-smooth-quant --batch-size 56 --calib-iters 100 --calib-shuffle --alpha 0.85 --output-dir {data['modelargs'][mode]['outputdir']}/facebook/opt-1.3b/  \
            #              2>&1 | tee -a $log_dir/llm_default_facebook-opt-1.3b_static8_autotune_SPR.log")
            lines.append("wait") 
            lines.append(f"python run.py -m EleutherAI/gpt-j-6b --ipex-smooth-quant --batch-size 56 --calib-iters 100 --calib-shuffle --fallback-add --alpha 0.85 --output-dir {data['modelargs'][mode]['outputdir']}/EleutherAI/gpt-j-6b/  \
                         2>&1 | tee -a $log_dir/llm_default_meta-llama-Llama-2-70b-hf_static8_autotune_SPR.log") 
            
            lines.append("wait")

        if mode.endswith('autotune3'):
            lines.append("# Run Workload") 
            lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/tiiuae/falcon-40b/")
            lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/baichuan-inc/Baichuan2-7B-Chat/")
            lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/baichuan-inc/Baichuan2-13B-Chat/")
            lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/THUDM/chatglm2-6b/")

            lines.append(f"python run.py -m tiiuae/falcon-40b --ipex-smooth-quant --batch-size 56 --calib-iters 100 --calib-shuffle --alpha 0.9 --output-dir {data['modelargs'][mode]['outputdir']}/tiiuae/falcon-40b/  \
                         2>&1 | tee -a $log_dir/llm_default_tiiuae-falcon-40b_static8_autotune_SPR.log")
            lines.append("wait")      
                  
            lines.append(f"python run.py -m baichuan-inc/Baichuan2-7B-Chat --ipex-smooth-quant --batch-size 56 --calib-iters 100 --calib-shuffle --alpha 0.95 --output-dir {data['modelargs'][mode]['outputdir']}/baichuan-inc/Baichuan2-7B-Chat/  \
                         2>&1 | tee -a $log_dir/llm_default_baichuan-inc-Baichuan2-7B-Chat_static8_autotune_SPR.log")
            lines.append("wait")
            lines.append(f"python run.py -m baichuan-inc/Baichuan2-13B-Chat --ipex-smooth-quant --batch-size 56 --calib-iters 100 --calib-shuffle --alpha 0.65 --output-dir {data['modelargs'][mode]['outputdir']}/baichuan-inc/Baichuan2-13B-Chat/  \
                         2>&1 | tee -a $log_dir/llm_default_baichuan-inc-Baichuan2-13B-Chat_static8_autotune_SPR.log")
            
            lines.append("wait")
            lines.append(f"python run.py -m THUDM/chatglm2-6b --ipex-smooth-quant --batch-size 56 --calib-iters 100 --calib-shuffle --alpha 0.75 --output-dir {data['modelargs'][mode]['outputdir']}/THUDM/chatglm2-6b/  \
                         2>&1 | tee -a $log_dir/llm_default_THUDM-chatglm2-6b_static8_autotune_SPR.log")
            
            lines.append("wait")

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

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                            
                                        if beam == True:   
                                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                        --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype {dtype} --batch-size {bs} --ipex --token-latency --profile   \
                                                            2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                        else:   
                                            if 'mpt' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype {dtype} --batch-size {bs} --ipex --token-latency --profile  --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")    
                                            elif 'falcon-40b' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype {dtype} --batch-size {bs} --ipex --token-latency --profile --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                  
                                            
                                            else:

                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype {dtype} --batch-size {bs} --ipex --token-latency --profile \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")



        if mode.endswith('inductor'):
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

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                            
                                        if beam == True:   
                                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python single_instance/run_generation.py  \
                                                        --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --torch-compile --backend inductor --profile   \
                                                            2>&1 | tee -a $log_dir/llm_inductor_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                        else:   
                                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python single_instance/run_generation.py  \
                                                        --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --torch-compile --backend inductor --profile \
                                                            2>&1 | tee -a $log_dir/llm_inductor_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_inductor_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")



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

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_compile_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                            
                                        if beam == True:   
                                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python single_instance/run_generation.py   \
                                                        --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype {dtype} --batch-size {bs} --ipex --token-latency --profile --torch-compile    \
                                                            2>&1 | tee -a $log_dir/llm_compile_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                        else:   
                                            if 'falcon-40b' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python single_instance/run_generation.py \
                                                            --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype {dtype} --batch-size {bs} --ipex --token-latency --profile --torch-compile --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                2>&1 | tee -a $log_dir/llm_compile_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log") 
                                            elif 'mpt' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python single_instance/run_generation.py \
                                                            --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype {dtype} --batch-size {bs} --ipex --token-latency --profile --torch-compile  --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                2>&1 | tee -a $log_dir/llm_compile_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                   
                                            
                                            else:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python single_instance/run_generation.py \
                                                            --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype {dtype} --batch-size {bs} --ipex --token-latency --profile --torch-compile   \
                                                                2>&1 | tee -a $log_dir/llm_compile_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_compile_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")



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

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{(model_id.replace('/','-')).replace('_','-')}_pt-{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                            
                                        if beam == True:   
                                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                        --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype {dtype} --batch-size {bs} --token-latency --profile  \
                                                            2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_pt-{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                        else:   
                                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                        --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype {dtype} --batch-size {bs} --token-latency --profile  \
                                                            2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_pt-{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{(model_id.replace('/','-')).replace('_','-')}_pt-{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")



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

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                        
                                        if 'fp32' in dtype:
                                            if beam == True:  
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-smooth-quant --qconfig-summary-file {data['modelargs'][mode]['configpath']}/{model_id}/best_configure.json --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency --profile    \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:  
                                                if 'falcon-40b' in model_id:
                                                    lines.append(f"python run.py  \
                                                                --benchmark -m {model_id} --ipex-smooth-quant --qconfig-summary-file {data['modelargs'][mode]['configpath']}/{model_id}/best_configure.json --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency --profile  --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log") 
                                                
                                                elif 'mpt' in model_id:
                                                    lines.append(f"python run.py  \
                                                                --benchmark -m {model_id} --ipex-smooth-quant --qconfig-summary-file {data['modelargs'][mode]['configpath']}/{model_id}/best_configure.json --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency --profile  --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                    
                                                
                                                else: 
                                                    lines.append(f"python run.py  \
                                                                --benchmark -m {model_id} --ipex-smooth-quant --qconfig-summary-file {data['modelargs'][mode]['configpath']}/{model_id}/best_configure.json --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency --profile     \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")       
                                        elif 'bf16' in dtype:
                                            if beam == True:   
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-smooth-quant --qconfig-summary-file {data['modelargs'][mode]['configpath']}/{model_id}/best_configure.json --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --batch-size {bs} --quant-with-amp --token-latency --profile    \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                if 'falcon-40b' in model_id:
                                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                                --benchmark -m {model_id} --ipex-smooth-quant --qconfig-summary-file {data['modelargs'][mode]['configpath']}/{model_id}/best_configure.json --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --quant-with-amp --token-latency --profile --config-file utils/model_config/tiiuae_falcon-40b_config.json   \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")        
                                                elif 'mpt' in model_id:
                                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                                --benchmark -m {model_id} --ipex-smooth-quant --qconfig-summary-file {data['modelargs'][mode]['configpath']}/{model_id}/best_configure.json --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --quant-with-amp --token-latency --profile  --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                   
                                                else:
                                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                                --benchmark -m {model_id} --ipex-smooth-quant --qconfig-summary-file {data['modelargs'][mode]['configpath']}/{model_id}/best_configure.json --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --quant-with-amp --token-latency --profile    \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                           
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")



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

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                        
                                        if 'fp32' in dtype:
                                            if beam == True:  
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-smooth-quant  --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency --profile  \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:  
                                                if 'falcon-40b' in model_id:
                                                    lines.append(f"OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  \
                                                                --benchmark -m {model_id} --ipex-smooth-quant --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency --profile --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log") 
                                                elif 'mpt' in model_id:
                                                    lines.append(f"python run.py  \
                                                                --benchmark -m {model_id} --ipex-smooth-quant --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency --profile  --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                      
                                                
                                                else: 
                                                    lines.append(f"python run.py  \
                                                                --benchmark -m {model_id} --ipex-smooth-quant --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency --profile   \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")       
                                        elif 'bf16' in dtype:
                                            if beam == True:   
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-smooth-quant --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --batch-size {bs} --quant-with-amp --token-latency --profile    \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-smooth-quant --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --quant-with-amp --token-latency --profile    \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                           
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
 
                                 

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

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                        
                                        if 'fp32' in dtype:
                                            if beam == True:  
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-smooth-quant --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency --profile     \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-smooth-quant --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency --profile     \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")       
                                          
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")


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
                                        2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}-{bs}_{data['launcher']['hw']}.log")
                            # elif 'bs1' in dtype:
                            #     lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}  -m {model_id} --dtype int8 --batch-size {bs} --ipex  --tasks lambada_openai \
                            #             2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}-{bs}_{data['launcher']['hw']}.log") 
                            else:
                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['outputdir']}/bf16/{model_id}/best_model.pt  -m {model_id} --dtype int8 --batch-size {bs} --quant-with-amp --ipex  --tasks lambada_openai \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}-{bs}_{data['launcher']['hw']}.log")                                         
            
                    lines.append(f"collect_acc_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}-{bs}_{data['launcher']['hw']}.log")





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

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                            
                                        if beam == True: 
                                            if 'falcon-40b' in model_id:  
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --weight-dtype INT8 --quant-with-amp --batch-size {bs} --token-latency --profile --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            elif 'neox' in model_id or 'dolly' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --weight-dtype INT8  --batch-size {bs} --token-latency --profile    \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                
                                            elif 'mpt' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --weight-dtype INT8 --quant-with-amp --batch-size {bs} --token-latency --profile --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                
                                            else:    
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --weight-dtype INT8 --quant-with-amp --batch-size {bs} --token-latency --profile    \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                        else:   
                                            if 'fp32' in dtype:

                                                if 'falcon-40b' in model_id:
                                                    lines.append(f"python run.py  \
                                                                --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --weight-dtype INT8 --batch-size {bs} --token-latency --profile --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                                elif 'neox' in model_id:
                                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py --group-size 256 \
                                                                --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --weight-dtype INT8  --batch-size {bs} --token-latency --profile   \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log") 
                                                elif 'dolly' in model_id:
                                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                                --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --weight-dtype INT8 --lowp-mode FP32 --batch-size {bs} --token-latency --profile   \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                  
                                                elif 'mpt' in model_id:
                                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                                --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --weight-dtype INT8 --batch-size {bs} --token-latency --profile  --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                   
                                                
                                                elif 'bloom' in model_id:
                                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py --group-size 128 \
                                                                --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --weight-dtype INT8 --batch-size {bs} --token-latency --profile   \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                      
                                                
                                                else:
                                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                                --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --weight-dtype INT8 --batch-size {bs} --token-latency --profile   \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  

                                            else:

                                                if 'falcon-40b' in model_id:
                                                    lines.append(f"python run.py  \
                                                                --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --weight-dtype INT8 --quant-with-amp --batch-size {bs} --token-latency --profile --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                                elif 'neox' in model_id or 'dolly' in model_id:
                                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                                --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --weight-dtype INT8  --batch-size {bs} --token-latency --profile    \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                  
                                                elif 'mpt' in model_id:
                                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                                --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --weight-dtype INT8 --quant-with-amp --batch-size {bs} --token-latency --profile  --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                   
                                                elif 'starcoder' in model_id:
                                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py --group-size 128 \
                                                                --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --weight-dtype INT8 --quant-with-amp --batch-size {bs} --token-latency --profile \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                       
                                                elif 'Baichuan2-13B' in model_id:
                                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py --group-size 64 \
                                                                --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --weight-dtype INT8 --quant-with-amp --batch-size {bs} --token-latency --profile \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                        
                                                else:
                                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                                --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --weight-dtype INT8 --quant-with-amp --batch-size {bs} --token-latency --profile \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")

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

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                            
                                        if beam == True:   
                                            if 'falcon-40b' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --gptq --weight-dtype INT4 --quant-with-amp --batch-size {bs} --token-latency --profile --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            elif 'neox' in model_id or 'dolly' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --gptq --weight-dtype INT4 --batch-size {bs} --token-latency --profile    \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                
                                            elif 'mpt' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --gptq --weight-dtype INT4 --quant-with-amp --batch-size {bs} --token-latency --profile  --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                            else:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --gptq --weight-dtype INT4 --quant-with-amp --batch-size {bs} --token-latency --profile    \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                        else:   
                                            if 'falcon-40b' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --gptq --weight-dtype INT4 --quant-with-amp --batch-size {bs} --token-latency --profile --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")    
                                            elif 'neox' in model_id or 'dolly' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --gptq --weight-dtype INT4 --batch-size {bs} --token-latency --profile    \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                 
                                            elif 'mpt' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --gptq --weight-dtype INT4 --quant-with-amp --batch-size {bs} --token-latency --profile  --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                 
                                            
                                            else:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --gptq --weight-dtype INT4 --quant-with-amp --batch-size {bs} --token-latency --profile    \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                           
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")


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

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                        if data['modelargs'][mode]['shard'] == True:
                                            if beam == True:   
                                                lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                            --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile --autotp  \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:  
                                                if 'bloom' in model_id:
                                                    lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp    \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                                else:

                                                    lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py \
                                                            --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile --autotp  \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                        else:

                                            if beam == True:   
                                                if 'falcon-40b' in model_id:
                                                    lines.append(f"timeout 77m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --shard-model --autotp   \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                else:
                                                    lines.append(f"timeout 87m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --shard-model --autotp     \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                if 'mpt' in model_id:
                                                    lines.append(f"timeout 47m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --shard-model --autotp    --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")   
                                                elif 'falcon-40b' in model_id:

                                                    lines.append(f"timeout 47m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --shard-model --autotp  --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                   
                                                
                                                elif 'bloom' in model_id:
                                                    lines.append(f"timeout 47m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile --autotp    \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                        
                                                else:

                                                    lines.append(f"timeout 47m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --shard-model --autotp     \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")


        if mode.endswith('bf16dsno'):
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

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                        if data['modelargs'][mode]['shard'] == True:
                                            if beam == True:   
                                                lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                            --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile --autotp  \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:  
                                                if 'bloom' in model_id:
                                                    lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp    \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                                else:

                                                    lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py \
                                                            --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile --autotp  \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                        else:

                                            if beam == True:   
                                                if 'falcon-40b' in model_id:
                                                    lines.append(f"timeout 77m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp   \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                else:
                                                    lines.append(f"timeout 87m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp     \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                if 'mpt' in model_id:
                                                    lines.append(f"timeout 47m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp    --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")   
                                                elif 'falcon-40b' in model_id:

                                                    lines.append(f"timeout 47m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp  --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                   
                                                
                                                elif 'bloom' in model_id:
                                                    lines.append(f"timeout 47m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp    \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                        
                                                else:

                                                    lines.append(f"timeout 47m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp     \
                                                                    2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")



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

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                        if data['modelargs'][mode]['shard'] == True:
                                            if beam == True:   
                                                lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                            --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile --autotp  \
                                                                2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py \
                                                            --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile --autotp  \
                                                                2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                        else:

                                            if beam == True:   
                                                lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                            --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp     \
                                                                2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                if 'mpt' in model_id:
                                                    lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp   --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  

                                                elif 'falcon-40b' in model_id:
                                                    lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp  --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                    2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                     
                                                else:

                                                    lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp     \
                                                                    2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")


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

                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                        if data['modelargs'][mode]['shard'] == True:
                                            if beam == True:   
                                                lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                            --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile --autotp  \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py \
                                                            --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile --autotp  \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                        else:

                                            if beam == True:   
                                                lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                            --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp     \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                            --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp     \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                          
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")


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

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                        if data['modelargs'][mode]['shard'] == True:
                                            if beam == True:   
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                        
                                                    else:

                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                else:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                        
                                                    elif 'falcon-40b' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    
                                                    else:

                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                          
                                                    
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                      
                                                    
                                                    else:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                                else:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                          
                                                    else:

                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                        else:

                                            if beam == True:   
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 77m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --shard-model --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 77m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    elif 'falcon-40b' in model_id:
                                                        lines.append(f"timeout 77m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --shard-model --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 77m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --shard-model --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                        
                                                
                                                    else:
                                                        lines.append(f"timeout 87m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --shard-model --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                else:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 77m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --shard-model --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    elif 'falcon-40b' in model_id:
                                                        lines.append(f"timeout 77m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --shard-model --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                        
                                                    
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 77m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                        
                                                    
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 77m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --shard-model --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                        
                                                    
                                                    else:
                                                        lines.append(f"timeout 77m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --shard-model --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id or 'dolly' in model_id or 'Baichuan-13B' in model_id or 'opt-30b' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --shard-model --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log") 
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --group-size 128 \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                          
                                                    elif 'falcon-40b' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --shard-model --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                      
                                                    elif 'starcoder' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --group-size 128 \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --shard-model --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                         
                                                    elif 'Baichuan2-13B-Chat' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --group-size 64 \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --shard-model --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                          
                                                    
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --shard-model --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                         
                                                    else:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --shard-model --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")      
                                                else:
                                                    
                                                    if 'neox' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --group-size 256 \
                                                                    --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --shard-model --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log") 
                                                        
                                                    elif 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --lowp-mode FP32 --batch-size {bs} --ipex --token-latency --profile  --shard-model --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log") 
                                                        
                                                    elif 'Baichuan-13B' in model_id or 'opt-30b' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --shard-model --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")     
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --group-size 128 \
                                                                    --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                      
                                                    elif 'falcon-40b' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                    --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --shard-model --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                          
                                                    
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --shard-model --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                           
                                                    
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --shard-model --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                            
                                                    elif 'starcoder' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --group-size 128 \
                                                                    --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --shard-model --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                                    elif 'Baichuan2-13B' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --group-size 64 \
                                                                    --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --shard-model --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                      
                                                    else:                                                      
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --shard-model --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")



        if mode.endswith('woqdsno'):
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

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                        if data['modelargs'][mode]['shard'] == True:
                                            if beam == True:   
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                        
                                                    else:

                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                else:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                        
                                                    elif 'falcon-40b' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    
                                                    else:

                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                          
                                                    
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                      
                                                    
                                                    else:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                                else:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                          
                                                    else:

                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                        else:

                                            if beam == True:   
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 77m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile   --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 77m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    elif 'falcon-40b' in model_id:
                                                        lines.append(f"timeout 77m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 77m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                        
                                                
                                                    else:
                                                        lines.append(f"timeout 87m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                else:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 77m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile   --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    elif 'falcon-40b' in model_id:
                                                        lines.append(f"timeout 77m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                        
                                                    
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 77m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                        
                                                    
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 77m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                        
                                                    
                                                    else:
                                                        lines.append(f"timeout 77m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id or 'dolly' in model_id or 'Baichuan-13B' in model_id or 'opt-30b' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile   --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log") 
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --group-size 128 \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                          
                                                    elif 'falcon-40b' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                      
                                                    elif 'starcoder' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --group-size 128 \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                         
                                                    elif 'Baichuan2-13B-Chat' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --group-size 64 \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                          
                                                    
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                         
                                                    else:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")      
                                                else:
                                                    
                                                    if 'neox' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --group-size 256 \
                                                                    --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile   --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log") 
                                                        
                                                    elif 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --lowp-mode FP32 --batch-size {bs} --ipex --token-latency --profile   --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log") 
                                                        
                                                    elif 'Baichuan-13B' in model_id or 'opt-30b' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile   --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")     
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --group-size 128 \
                                                                    --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                      
                                                    elif 'falcon-40b' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                    --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                          
                                                    
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                           
                                                    
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                            
                                                    elif 'starcoder' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --group-size 128 \
                                                                    --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                                    elif 'Baichuan2-13B' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --group-size 64 \
                                                                    --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                      
                                                    else:                                                      
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id}  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")




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

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                        if data['modelargs'][mode]['shard'] == True:
                                            if beam == True:   
                                                if 'int4' in dtype:
                                                    if 'falcon-40b' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                        
                                                    elif 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    else:

                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                else:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    else:

                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                if 'int4' in dtype: 
                                                    if 'falcon-40b' in model_id: 
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                                    elif 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                        
                                                    
                                                    else:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                                else:
                                                    if 'falcon-40b' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                                    elif 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                          
                                                    
                                                    else:

                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                        else:

                                            if beam == True:   
                                                if 'int4' in dtype:
                                                    if 'falcon-40b' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --autotp  --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    elif 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    else:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                else:
                                                    
                                                    if 'falcon-40b' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --autotp  --config-file utils/model_config/tiiuae_falcon-40b_config.json   \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                        
                                                    
                                                    elif 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    else:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                if 'int4' in dtype:
                                                    if 'falcon-40b' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --autotp  --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                          
                                                    elif 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log") 
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                          
                                                    
                                                    else:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")      
                                                else:
                                                    if 'falcon-40b' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --autotp  --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                          
                                                    elif 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")     
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                     
                                                    else:                                                      
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_mode-p_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")


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

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                        if data['modelargs'][mode]['shard'] == True:
                                            if beam == True:   
                                                lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                            --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile --autotp  \
                                                                2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py \
                                                            --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile --autotp  \
                                                                2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                        else:

                                            if beam == True:   
                                                lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                            --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp    \
                                                                2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            elif 'mpt' in model_id:
                                                lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                            --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp  \
                                                                2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                  
                                            
                                            else:   
                                                lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                            --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp    \
                                                                2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")




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

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                        if data['modelargs'][mode]['shard'] == True:
                                            if beam == True:   
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    else:

                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                else:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    else:

                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                if 'int4' in dtype:
                                                    if 'falcon-40b' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                                    elif 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    else:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                                else:
                                                    if 'falcon-40b' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                         
                                                    elif 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                                    else:

                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                        else:

                                            if beam == True:   
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    else:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                else:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    else:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                if 'int4' in dtype:
                                                    if 'falcon-40b' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --autotp  --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log") 
                                                    elif 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log") 
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                          
                                                    
                                                    else:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")      
                                                else:
                                                    if 'falcon-40b' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --autotp --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                           
                                                    elif 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")     
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --autotp \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                            
                                                    
                                                    else:                                                      
                                                        lines.append(f"timeout 97m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_mode-m_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")




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
                if 'falcon-40b' in model_id: 
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
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                        
                        elif 'falcon-40b' in model_id:
                            lines.append(f"python single_instance/run_accuracy.py -m {model_id} --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt --batch-size 1 \
                                            --dtype int8   --quant-with-amp --tasks lambada_openai --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                        
                        elif 'neox' in model_id or 'dolly' in model_id:
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python single_instance/run_accuracy.py -m {model_id} --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt --batch-size 1 \
                                            --dtype int8   --tasks lambada_openai \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        elif 'mpt' in model_id:
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python single_instance/run_accuracy.py -m {model_id} --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt --batch-size 1 \
                                            --dtype int8   --quant-with-amp --tasks hellaswag --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        else:
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python single_instance/run_accuracy.py -m {model_id} --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt --batch-size 1 \
                                            --dtype int8   --quant-with-amp --tasks lambada_openai \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")

                        if 'codegen' in model_id in model_id or 'mpt' in model_id:
                            lines.append(f"collect_accnorm_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        else:
                            lines.append(f"collect_acc_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")


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
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python {data['modelargs'][mode]['scriptname']} \
                                                            -m {model_id} --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt --benchmark --greedy --quant-with-amp --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --ipex-weight-only-quantization --weight-dtype INT4 --token-latency --profile  \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                        else:
                                            if 'neox' in model_id or 'dolly' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python {data['modelargs'][mode]['scriptname']} \
                                                            -m {model_id} --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt --benchmark  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --ipex-weight-only-quantization --weight-dtype INT4 --token-latency --profile  \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            elif 'falcon-40b' in model_id: 

                                                lines.append(f"python {data['modelargs'][mode]['scriptname']} \
                                                            -m {model_id} --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt --benchmark --quant-with-amp --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --ipex-weight-only-quantization --weight-dtype INT4 --token-latency --profile --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                            elif 'mpt' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python {data['modelargs'][mode]['scriptname']} --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                            -m {model_id} --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt --benchmark --quant-with-amp --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --ipex-weight-only-quantization --weight-dtype INT4 --token-latency --profile  \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                
                                            
                                            else:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python {data['modelargs'][mode]['scriptname']} \
                                                            -m {model_id} --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt --benchmark --quant-with-amp --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --ipex-weight-only-quantization --weight-dtype INT4 --token-latency --profile  \
                                                                2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                        
                                        lines.append(f"collect_perf_logs_llm llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")


        if mode.endswith('defaultacc'):
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for rank in data['modelargs'][mode]['localrank']:
                        lines.append(f"export local_rank={rank}")
                        lines.append("deepspeed_core_config ${local_rank}")
                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")
                        # if 'codegen' in model_id:
                        #     lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py  -m {model_id} --dtype {dtype} --ipex  --tasks hellaswag --batch-size 1\
                        #                 2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        # elif 'mpt' in model_id:
                        #     lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py  -m {model_id} --dtype {dtype} --ipex  --tasks lambada_openai --batch-size 1 --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                        #                 2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        # else:
                        #     lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py  -m {model_id} --dtype {dtype} --ipex  --tasks lambada_openai --batch-size 1 \
                        #                 2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                            
                        if 'codegen' in model_id:
                            lines.append(f"python single_instance/run_accuracy.py  -m {model_id} --dtype {dtype} --ipex  --tasks hellaswag --batch-size 1 \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        elif 'falcon-40b' in model_id:
                            lines.append(f"python single_instance/run_accuracy.py  -m {model_id} --dtype {dtype} --ipex  --tasks lambada_openai --batch-size 1 --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        elif 'mpt' in model_id:
                            lines.append(f"python single_instance/run_accuracy.py  -m {model_id} --dtype {dtype} --ipex  --tasks hellaswag --batch-size 1 --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                                                         
                        else:
                            lines.append(f"python single_instance/run_accuracy.py  -m {model_id} --dtype {dtype} --ipex  --tasks lambada_openai --batch-size 1 \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        if 'codegen' in model_id or 'mpt' in model_id:
                            lines.append(f"collect_accnorm_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        else:
                            lines.append(f"collect_acc_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")

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
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                            elif 'mpt' in model_id:
                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py  -m {model_id} --dtype {dtype} --ipex  --tasks {task} --batch-size 1 --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{task}_{data['launcher']['hw']}.log")                                
                            
                            else:
                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py  -m {model_id} --dtype {dtype} --ipex  --tasks {task} --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{task}_{data['launcher']['hw']}.log")
                            if 'codegen' in model_id or 'mpt' in model_id:
                                lines.append(f"collect_accnorm_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{task}_{data['launcher']['hw']}.log")
                            else:
                                lines.append(f"collect_acc_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{task}_{data['launcher']['hw']}.log")


        if mode.endswith('defaultaccnocore'):
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for rank in data['modelargs'][mode]['localrank']:
                        lines.append(f"export local_rank={rank}")
                        lines.append("deepspeed_core_config ${local_rank}")
                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")
                        if 'codegen' in model_id:
                            lines.append(f"python single_instance/run_accuracy.py  -m {model_id} --dtype {dtype} --ipex  --tasks hellaswag --batch-size 56\
                                        2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        else:
                            lines.append(f"python single_instance/run_accuracy.py  -m {model_id} --dtype {dtype} --ipex  --tasks lambada_openai --batch-size 56\
                                        2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        if 'codegen' in model_id or 'mpt' in model_id:
                            lines.append(f"collect_accnorm_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        else:
                            lines.append(f"collect_acc_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")

        if mode.endswith('static8acc'):
            for model_id in data['modelargs'][mode]['modelid']:
                for rank in data['modelargs'][mode]['localrank']:
                    lines.append(f"export local_rank={rank}")
                    lines.append("deepspeed_core_config ${local_rank}")
                    lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")
                    for dtype in data['modelargs'][mode]['dtype']:
                        for bs in data['modelargs'][mode]['batchsize']:
                            if 'fp32' in dtype:
                                if 'falcon-40b' in model_id: 
                                    lines.append(f"python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --batch-size {bs} --ipex --tasks lambada_openai --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                                elif 'mpt' in model_id:
                                    lines.append(f"python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --batch-size {bs} --ipex --tasks hellaswag --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                                    
                                elif 'codegen' in model_id:
                                    lines.append(f"python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --batch-size {bs} --ipex --tasks hellaswag \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                                
                                else:
                                    lines.append(f"python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --batch-size {bs} --ipex --tasks lambada_openai \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                            # elif 'bs1' in dtype:
                            #     lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}  -m {model_id} --dtype int8 --batch-size {bs} --ipex  --tasks lambada_openai \
                            #             2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}-{bs}_{data['launcher']['hw']}.log") 
                            else:
                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --batch-size {bs} --quant-with-amp --ipex --tasks lambada_openai \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                                
                            if 'codegen' in model_id in model_id or 'mpt' in model_id:
                                lines.append(f"collect_accnorm_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                            else:
                                lines.append(f"collect_acc_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")



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
                                                2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                                elif 'neox' in model_id or 'dolly' in model_id:
                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --ipex  --tasks lambada_openai --batch-size {bs} \
                                                2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                                elif 'falcon-40b' in model_id:
                                    lines.append(f"python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --ipex  --tasks lambada_openai --batch-size {bs} --config-file utils/model_config/tiiuae_falcon-40b_config.json\
                                                2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                                elif 'mpt' in model_id:
                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --ipex  --tasks hellaswag --batch-size {bs} --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                                                2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                                
                                # elif 'dolly' in model_id:
                                #     lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --quant-with-amp --ipex  --tasks hellaswag --batch-size {bs} \
                                #                 2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                               
                                else:
                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --ipex --tasks lambada_openai --batch-size {bs} \
                                                2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")

                            else:
                                if 'codegen' in model_id:                             
                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --ipex --quant-with-amp --tasks hellaswag --batch-size {bs} \
                                                2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                                elif 'neox' in model_id or 'dolly' in model_id:
                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --ipex  --tasks lambada_openai --batch-size {bs}  \
                                                2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                                elif 'falcon-40b' in model_id:
                                    lines.append(f"python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --quant-with-amp --ipex  --tasks lambada_openai --batch-size {bs} --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                                elif 'mpt' in model_id:
                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --quant-with-amp --ipex  --tasks hellaswag --batch-size {bs} --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                                                2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                                
                                # elif 'dolly' in model_id:
                                #     lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --quant-with-amp --ipex  --tasks hellaswag --batch-size {bs} \
                                #                 2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                               
                                else:
                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --ipex --quant-with-amp --tasks lambada_openai --batch-size {bs} \
                                                2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                            
                            if 'codegen' in model_id in model_id or 'mpt' in model_id:
                                lines.append(f"collect_accnorm_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                            else:
                                lines.append(f"collect_acc_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                            



        if mode.endswith('woq8acc1'):
            for model_id in data['modelargs'][mode]['modelid']:
                for rank in data['modelargs'][mode]['localrank']:
                    lines.append(f"export local_rank={rank}")
                    lines.append("deepspeed_core_config ${local_rank}")
                    lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")
                    for dtype in data['modelargs'][mode]['dtype']:
                        if 'codegen' in model_id:                             
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}  -m {model_id} --dtype int8 --quant-with-amp --ipex  --tasks hellaswag --batch-size 56 \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        elif 'neox' in model_id or 'dolly' in model_id:
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}  -m {model_id} --dtype int8 --ipex  --tasks lambada_openai --batch-size 56 \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        elif 'falcon-40b' in model_id:
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}  -m {model_id} --dtype int8 --quant-with-amp --ipex  --tasks lambada_openai --batch-size 56 \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        else:
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}  -m {model_id} --dtype int8 --quant-with-amp --ipex  --tasks lambada_openai --batch-size 56 \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        if 'codegen' in model_id or 'mpt' in model_id:
                            lines.append(f"collect_accnorm_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        else:
                            lines.append(f"collect_acc_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")


        if mode.endswith('bf16dsacc'):
            # lines.append("cd ./distributed")
            lines.append("unset KMP_AFFINITY")
            for model_id in data['modelargs'][mode]['modelid']:
                if 'falcon-40b' in model_id: 
                    lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --ipex  --tasks lambada_openai --batch-size 1 --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                    2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_ds-bfloat16_{data['launcher']['hw']}.log")                
                elif 'codegen' in model_id:
                    lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --ipex  --tasks hellaswag  --batch-size 1 \
                                    2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_ds-bfloat16_{data['launcher']['hw']}.log")                    
                elif 'gpt-j' in model_id:
                    lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --ipex  --tasks lambada_openai  --batch-size 1 \
                                    2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_ds-bfloat16_{data['launcher']['hw']}.log")                    
                elif 'mpt' in model_id:
                    lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --ipex  --tasks hellaswag  --batch-size 1 --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                                    2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_ds-bfloat16_{data['launcher']['hw']}.log")                    
                # elif 'dolly' in model_id:
                #     lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --ipex  --tasks hellaswag  --batch-size 1 \
                #                     2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_ds-bfloat16_{data['launcher']['hw']}.log")                  
                else:
                    lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --ipex  --tasks lambada_openai  --batch-size 1 \
                                    2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_ds-bfloat16_{data['launcher']['hw']}.log")
                
                if 'mpt' in model_id or 'codegen' in model_id:
                    lines.append(f"collect_accnorm_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_ds-bfloat16_{data['launcher']['hw']}.log")
                else:
                    lines.append(f"collect_acc_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_ds-bfloat16_{data['launcher']['hw']}.log")

        if mode.endswith('bf16dsaccnoipex'):
            # lines.append("cd ./distributed")
            lines.append("unset KMP_AFFINITY")
            for model_id in data['modelargs'][mode]['modelid']:
                if 'falcon-40b' in model_id: 
                    lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --tasks lambada_openai --batch-size 1 --disable-jit \
                                    2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_ds-bfloat16_{data['launcher']['hw']}.log")                
                elif 'codegen' in model_id:
                    lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --tasks hellaswag  --batch-size 1 --disable-jit \
                                    2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_ds-bfloat16_{data['launcher']['hw']}.log")                    
                elif 'gpt-j' in model_id:
                    lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --tasks lambada_openai  --batch-size 1 --disable-jit \
                                    2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_ds-bfloat16_{data['launcher']['hw']}.log")                    
                elif 'mpt' in model_id:
                    lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --tasks hellaswag --disable-jit --batch-size 1 --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                                    2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_ds-bfloat16_{data['launcher']['hw']}.log")                    
                # elif 'dolly' in model_id:
                #     lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --ipex  --tasks hellaswag  --batch-size 1 \
                #                     2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_ds-bfloat16_{data['launcher']['hw']}.log")                  
                else:
                    lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --tasks lambada_openai --disable-jit --batch-size 1 \
                                    2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_ds-bfloat16_{data['launcher']['hw']}.log")
                
                if 'mpt' in model_id or 'codegen' in model_id:
                    lines.append(f"collect_accnorm_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_ds-bfloat16_{data['launcher']['hw']}.log")
                else:
                    lines.append(f"collect_acc_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_ds-bfloat16_{data['launcher']['hw']}.log")

        if mode.endswith('woqdsacc'):

            lines.append("unset KMP_AFFINITY")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    if 'int8' in dtype:
                        if 'neox' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --dtype float32 --ipex  --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 --group-size 256 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                            
                        elif 'dolly' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --dtype float32 --ipex  --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 --lowp-mode FP32 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        elif 'Baichuan-13B' in model_id or 'opt-30b' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --dtype float32 --ipex  --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        elif 'falcon-40b' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --quant-with-amp --ipex  --tasks lambada_openai  --ipex-weight-only-quantization  --batch-size 1 --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        elif 'starcoder' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --quant-with-amp --ipex  --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        
                        elif 'codegen' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --quant-with-amp --ipex  --tasks hellaswag  --ipex-weight-only-quantization --batch-size 1 --group-size 128 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        elif 'mpt' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --quant-with-amp --ipex  --tasks hellaswag  --ipex-weight-only-quantization --batch-size 1 --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        # elif 'dolly' in model_id:
                        #     lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --quant-with-amp --ipex  --tasks hellaswag  --ipex-weight-only-quantization --batch-size 1 \
                        #                     2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                           
                        else:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --quant-with-amp --ipex  --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        if 'codegen' in model_id or 'mpt' in model_id:
                            lines.append(f"collect_accnorm_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        else:
                            lines.append(f"collect_acc_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")

                    elif 'int4' in dtype:
                        if 'neox' in model_id or 'dolly' in model_id or 'Baichuan-13B' in model_id or 'opt-30b' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --dtype float32 --ipex  --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        elif 'falcon-40b' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --quant-with-amp --ipex  --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        elif 'codegen' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --quant-with-amp --ipex  --tasks hellaswag  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        elif 'mpt' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --quant-with-amp --ipex  --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        # elif 'dolly' in model_id:
                        #     lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --quant-with-amp --ipex  --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 \
                        #                     2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                           
                        else:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --quant-with-amp --ipex  --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        if 'codegen' in model_id or 'mpt' in model_id:
                            lines.append(f"collect_accnorm_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        else:
                            lines.append(f"collect_acc_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")


        if mode.endswith('woqdsaccnoipex'):

            lines.append("unset KMP_AFFINITY")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    if 'int8' in dtype:
                        if 'neox' in model_id or 'dolly' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --dtype float32 --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        elif 'falcon-40b' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --quant-with-amp --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        elif 'codegen' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --quant-with-amp --tasks hellaswag  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        elif 'mpt' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --quant-with-amp --tasks hellaswag  --ipex-weight-only-quantization --batch-size 1 --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        # elif 'dolly' in model_id:
                        #     lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --quant-with-amp --tasks hellaswag  --ipex-weight-only-quantization --batch-size 1 \
                        #                     2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                           
                        else:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --quant-with-amp --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        if 'codegen' in model_id or 'mpt' in model_id:
                            lines.append(f"collect_accnorm_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        else:
                            lines.append(f"collect_acc_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")

                    elif 'int4' in dtype:
                        if 'neox' in model_id or 'dolly' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --dtype float32 --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        elif 'falcon-40b' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --quant-with-amp --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        elif 'codegen' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --quant-with-amp --tasks hellaswag  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        elif 'mpt' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --quant-with-amp --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        # elif 'dolly' in model_id:
                        #     lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --quant-with-amp --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 \
                        #                     2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")                           
                        else:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --quant-with-amp --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        if 'codegen' in model_id or 'mpt' in model_id:
                            lines.append(f"collect_accnorm_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")
                        else:
                            lines.append(f"collect_acc_logs_llm llm_accuracy_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{data['launcher']['hw']}.log")


        
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
                for dtype in data['modelargs'][mode]['dtype']:
                        if 'bf16' in dtype:
                            if 'falcon-40b' in model_id: 
                                lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --quant-with-amp --lowp-mode BF16 -m {model_id} --low-precision-checkpoint {data['modelargs'][mode]['configpath']}/{model_id}/gptq_checkpoint_g128.pt --config-file utils/model_config/tiiuae_falcon-40b_config.json")
                            elif 'neox' in model_id or 'dolly' in model_id:
                                lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id}  -m {model_id} --lowp-mode BF16 --low-precision-checkpoint {data['modelargs'][mode]['configpath']}/{model_id}/gptq_checkpoint_g128.pt")                    
                            elif 'mpt' in model_id:
                                lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --quant-with-amp --lowp-mode BF16 -m {model_id} --low-precision-checkpoint {data['modelargs'][mode]['configpath']}/{model_id}/gptq_checkpoint_g128.pt --config-file=utils/model_config/mosaicml_mpt-7b_config.json")                    
                            
                            else:
                                lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --quant-with-amp --lowp-mode BF16 -m {model_id} --low-precision-checkpoint {data['modelargs'][mode]['configpath']}/{model_id}/gptq_checkpoint_g128.pt")
                        else:
                            if 'falcon-40b' in model_id: 
                                lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --quant-with-amp --lowp-mode INT8 -m {model_id} --low-precision-checkpoint {data['modelargs'][mode]['configpath']}/{model_id}/gptq_checkpoint_g128.pt --config-file utils/model_config/tiiuae_falcon-40b_config.json")
                            elif 'neox' in model_id or 'dolly' in model_id:
                                lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id}  -m {model_id} --lowp-mode INT8 --low-precision-checkpoint {data['modelargs'][mode]['configpath']}/{model_id}/gptq_checkpoint_g128.pt")                    
                            elif 'mpt' in model_id:
                                lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --quant-with-amp --lowp-mode INT8 -m {model_id} --low-precision-checkpoint {data['modelargs'][mode]['configpath']}/{model_id}/gptq_checkpoint_g128.pt --config-file=utils/model_config/mosaicml_mpt-7b_config.json")                    
                            
                            else:
                                lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --quant-with-amp --lowp-mode INT8 -m {model_id} --low-precision-checkpoint {data['modelargs'][mode]['configpath']}/{model_id}/gptq_checkpoint_g128.pt")




   
        lines.append(f"sleep 5s")
        lines.append("")
        runfile.writelines([line + "\n" for line in lines])
    return generated_file


if __name__ == '__main__':
    #for mode in 'default','gptj_int8','llama_int8','deepspeed':
    yml_file = 'bench_preci.yml'
    if args.acc_cluster1:    
        yml_file = 'bench_acc_cluster1.yml'
    if args.acc_cluster2:    
        yml_file = 'bench_acc_cluster2.yml'
    if args.acc_cluster3:    
        yml_file = 'bench_acc_cluster3.yml'
    if args.acc_cluster4:    
        yml_file = 'bench_acc_cluster4.yml'
    data = yaml.load(open(yml_file, 'r'),Loader=yaml.FullLoader) 
    for mode in data['modelargs'].keys():
        generate_commands(yml_file, mode, args.extra_kmp)
