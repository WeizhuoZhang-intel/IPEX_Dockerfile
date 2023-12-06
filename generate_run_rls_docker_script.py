# Generate runnable scripts for benchmarking
import argparse
import yaml
parser = argparse.ArgumentParser("Generation script", add_help=False)
parser.add_argument("-k","--extra_kmp",action="store_true",default=False,help="llm extra kmp configuration")
parser.add_argument("-d","--deepspeed",action="store_true",default=False,help="only for deepspeed")
parser.add_argument("--nightly",action="store_true",default=False,help="only for nightly regular track")
parser.add_argument("--weekly",action="store_true",default=False,help="only for weekly regular track")
parser.add_argument("--emr_weekly",action="store_true",default=False,help="only for weekly regular track")
parser.add_argument("--hbm_weekly",action="store_true",default=False,help="only for weekly regular track")
parser.add_argument("--hbm_nightly",action="store_true",default=False,help="only for weekly regular track")
parser.add_argument("--gnr_weekly",action="store_true",default=False,help="only for weekly regular track")
parser.add_argument("--debug",action="store_true",default=False,help="only for debug regular track")
parser.add_argument("--rls",action="store_true",default=False,help="only for rls track")
parser.add_argument("--acc",action="store_true",default=False,help="only for rls track")
parser.add_argument("--rlsemr",action="store_true",default=False,help="only for rls track")
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
        lines.append("export WORKDIR=/root/workspace/llm")
        lines.append("export HF_HOME=/root/.cache/huggingface")
        lines.append("export TRANSFORMERS_OFFLINE=0")
        lines.append("pip install --upgrade huggingface_hub")
        lines.append("huggingface-cli login --token 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")
        lines.append("log_dir=/root/workspace/log")

        lines.append("# device info")
        lines.append(fetch_device_info)
        lines.append(collect_result)    
        lines.append(deepspeed_ccl_func)
        lines.append("")
        lines.append("cp prompt.json ./single_instance") 

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
                                                        --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype {dtype} --batch-size {bs} --ipex --deployment-mode --token-latency   \
                                                            2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                        else:   
                                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                        --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype {dtype} --batch-size {bs} --ipex --deployment-mode --token-latency   \
                                                            2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")

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
                                                        --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype {dtype} --batch-size {bs} --token-latency\
                                                            2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_pt-{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                        else:   
                                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                        --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype {dtype} --batch-size {bs} --token-latency\
                                                            2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_pt-{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_pt-{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")



        if mode.endswith('static8'):
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
                                                            --benchmark -m {model_id} --ipex-smooth-quant --qconfig-summary-file {data['modelargs'][mode]['configpath']} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --batch-size {bs} --int8 --token-latency   \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-smooth-quant --qconfig-summary-file {data['modelargs'][mode]['configpath']} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --int8 --token-latency   \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")       
                                        elif 'bf16' in dtype:
                                            if beam == True:   
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-smooth-quant --qconfig-summary-file {data['modelargs'][mode]['configpath']} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --batch-size {bs} --int8-bf16-mixed --token-latency   \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-smooth-quant --qconfig-summary-file {data['modelargs'][mode]['configpath']} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --int8-bf16-mixed --token-latency   \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                           
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")


        if mode.endswith('woq8'):
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
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --int8-bf16-mixed --batch-size {bs} --token-latency --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            elif 'neox' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --int8 --batch-size {bs} --token-latency   \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                
                                            else:    
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --int8-bf16-mixed --batch-size {bs} --token-latency   \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                        else:   
                                            if 'falcon' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --int8-bf16-mixed --batch-size {bs} --token-latency --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                            elif 'neox' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --int8 --batch-size {bs} --token-latency   \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                  
                                            elif 'falcon' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --int8-bf16-mixed --batch-size {bs} --token-latency   \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")

        if mode.endswith('woq4'):
            lines.append("# Run Workload")  
            lines.append("export WORK_DIR=./")
            for model_id in data['modelargs'][mode]['modelid']:
                # lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}")
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
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --gptq --int8-bf16-mixed --batch-size {bs} --token-latency --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            elif 'neox' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --gptq --int8 --batch-size {bs} --token-latency   \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                
                                            else:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --gptq --int8-bf16-mixed --batch-size {bs} --token-latency   \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                        else:   
                                            if 'falcon' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --gptq --int8-bf16-mixed --batch-size {bs} --token-latency --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")    
                                            elif 'neox' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --gptq --int8 --batch-size {bs} --token-latency   \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                 
                                            else:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --gptq --int8-bf16-mixed --batch-size {bs} --token-latency   \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                           
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")


        if mode.endswith('bf16ds'):
            lines.append("# Run Workload") 
            lines.append("cp prompt.json ./distributed") 
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
                                                            --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --deployment-mode --token-latency --autotp  \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py \
                                                            --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --deployment-mode --token-latency --autotp  \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                        else:

                                            if beam == True:   
                                                lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                            --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --deployment-mode --autotp --shard-model --token-latency   \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                            --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --deployment-mode --autotp --shard-model --token-latency   \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")

        if mode.endswith('woqds'):
            lines.append("# Run Workload") 
            lines.append("cp prompt.json ./distributed") 
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
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id:
                                                        lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --deployment-mode --batch-size {bs} --weight-dtype INT4 --lowp-mode INT8 --ipex --int8 --token-latency --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    else:

                                                        lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --deployment-mode --batch-size {bs} --weight-dtype INT4 --lowp-mode INT8 --ipex --int8-bf16-mixed --token-latency --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                else:
                                                    if 'neox' in model_id:
                                                        lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --deployment-mode --batch-size {bs} --ipex --int8 --token-latency --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    else:

                                                        lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --deployment-mode --batch-size {bs} --ipex --int8-bf16-mixed --token-latency --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id:
                                                        lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --deployment-mode --batch-size {bs} --weight-dtype INT4 --lowp-mode INT8 --ipex --int8 --token-latency --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    else:
                                                        lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --deployment-mode --batch-size {bs} --weight-dtype INT4 --lowp-mode INT8 --ipex --int8-bf16-mixed --token-latency --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                                else:
                                                    if 'neox' in model_id:
                                                        lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --deployment-mode --batch-size {bs} --ipex --int8 --token-latency --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                                    else:

                                                        lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --deployment-mode --batch-size {bs} --ipex --int8-bf16-mixed --token-latency --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                        else:

                                            if beam == True:   
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id:
                                                        lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --deployment-mode --batch-size {bs} --weight-dtype INT4 --lowp-mode INT8 --ipex --int8 --autotp --shard-model --token-latency   \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    else:
                                                        lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --deployment-mode --batch-size {bs} --weight-dtype INT4 --lowp-mode INT8 --ipex --int8-bf16-mixed --autotp --shard-model --token-latency   \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                else:
                                                    if 'neox' in model_id:
                                                        lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --deployment-mode --batch-size {bs} --ipex --int8 --autotp --shard-model --token-latency   \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    else:
                                                        lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --deployment-mode --batch-size {bs} --ipex --int8-bf16-mixed --autotp --shard-model --token-latency   \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id:
                                                        lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --deployment-mode --batch-size {bs} --weight-dtype INT4 --lowp-mode INT8 --ipex --int8 --autotp --shard-model --token-latency   \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log") 
                                                    else:
                                                        lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --deployment-mode --batch-size {bs} --weight-dtype INT4 --lowp-mode INT8 --ipex --int8-bf16-mixed --autotp --shard-model --token-latency   \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")      
                                                else:
                                                    if 'neox' in model_id:
                                                        lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --deployment-mode --batch-size {bs} --ipex --int8 --autotp --shard-model --token-latency   \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")     
                                                    else:                                                      
                                                        lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --deployment-mode --batch-size {bs} --ipex --int8-bf16-mixed --autotp --shard-model --token-latency   \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")


        if mode.endswith('gptq'):
            lines.append("# Run Workload")  
            lines.append("export WORK_DIR=./")
            # lines.append("unset KMP_BLOCKTIME KMP_TPAUSE KMP_SETTINGS KMP_AFFINITY KMP_FORJOIN_BARRIER_PATTERN KMP_PLAIN_BARRIER_PATTERN KMP_REDUCTION_BARRIER_PATTERN")
            for model_id in data['modelargs'][mode]['modelid']:
                lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}")
                lines.append("ls utils")
                lines.append("pwd")
                lines.append(f"python utils/run_gptq.py --model {model_id} --output-dir {data['modelargs'][mode]['outputdir']}")
                lines.append("wait")
                if 'falcon' in model_id: 
                    lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']} --int8-bf16-mixed -m {model_id} --low-precision-checkpoint {data['modelargs'][mode]['gptqpath']} --config-file utils/model_config/tiiuae_falcon-40b_config.json")
                elif 'neox' in model_id:
                    lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']} --int8 -m {model_id} --low-precision-checkpoint {data['modelargs'][mode]['gptqpath']}")                    
                else:
                    lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']} --int8-bf16-mixed -m {model_id} --low-precision-checkpoint {data['modelargs'][mode]['gptqpath']}")

        if mode.endswith('gptqacc'):
            lines.append("# Run Workload")  
            lines.append("export WORK_DIR=./")
            # lines.append("source /tools/env_activate.sh")
            for model_id in data['modelargs'][mode]['modelid']:
                for rank in data['modelargs'][mode]['localrank']:
                    lines.append(f"export local_rank={rank}")
                    lines.append("deepspeed_core_config ${local_rank}")
                    lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")                    
                    lines.append("ls utils")
                    lines.append("pwd")
                    # lines.append(f"python utils/run_gptq.py --model {model_id} --output-dir {data['modelargs'][mode]['outputdir']}")
                    # lines.append("wait")
                    if 'neox' in model_id:
                        lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python single_instance/run_accuracy.py -m {model_id} --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --batch-size 56 \
                                        --dtype int8 --accuracy-only --jit --int8 --tasks lambada_openai \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_woq-int4_{data['launcher']['hw']}.log")
                    else:
                        lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python single_instance/run_accuracy.py -m {model_id} --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --batch-size 56 \
                                        --dtype int8 --accuracy-only --jit --int8-bf16-mixed --tasks lambada_openai \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_woq-int4_{data['launcher']['hw']}.log")


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
                                            if 'neox' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python {data['modelargs'][mode]['scriptname']} \
                                                            -m {model_id} --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --benchmark --greedy --int8 --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python {data['modelargs'][mode]['scriptname']} \
                                                            -m {model_id} --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --benchmark --greedy --int8-bf16-mixed --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                        else:
                                            if 'neox' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python {data['modelargs'][mode]['scriptname']} \
                                                            -m {model_id} --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --benchmark --int8 --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python {data['modelargs'][mode]['scriptname']} \
                                                            -m {model_id} --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --benchmark --int8-bf16-mixed --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                        
                                        lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")


        if mode.endswith('defaultacc'):
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for rank in data['modelargs'][mode]['localrank']:
                        lines.append(f"export local_rank={rank}")
                        lines.append("deepspeed_core_config ${local_rank}")
                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")
                        if 'codegen' in model_id:
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --accuracy-only -m {model_id} --dtype {dtype} --ipex --jit --tasks hellaswag --batch-size 56\
                                        2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        else:
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --accuracy-only -m {model_id} --dtype {dtype} --ipex --jit --tasks lambada_openai --batch-size 56 \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
        if mode.endswith('defaultaccnocore'):
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for rank in data['modelargs'][mode]['localrank']:
                        lines.append(f"export local_rank={rank}")
                        lines.append("deepspeed_core_config ${local_rank}")
                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")
                        if 'codegen' in model_id:
                            lines.append(f"python single_instance/run_accuracy.py --accuracy-only -m {model_id} --dtype {dtype} --ipex --jit --tasks hellaswag --batch-size 56\
                                        2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        else:
                            lines.append(f"python single_instance/run_accuracy.py --accuracy-only -m {model_id} --dtype {dtype} --ipex --jit --tasks lambada_openai --batch-size 56\
                                        2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")


        if mode.endswith('static8acc'):
            for model_id in data['modelargs'][mode]['modelid']:
                for rank in data['modelargs'][mode]['localrank']:
                    lines.append(f"export local_rank={rank}")
                    lines.append("deepspeed_core_config ${local_rank}")
                    lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")
                    for dtype in data['modelargs'][mode]['dtype']:
                        if 'fp32' in dtype:
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --accuracy-only -m {model_id} --dtype int8 --batch-size 56 --ipex --jit --tasks lambada_openai \
                                    2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}-56_{data['launcher']['hw']}.log")
                        elif 'bs1' in dtype:
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --accuracy-only -m {model_id} --dtype int8 --batch-size 1 --ipex --jit --tasks lambada_openai \
                                    2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}-1_{data['launcher']['hw']}.log") 
                        else:
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --accuracy-only -m {model_id} --dtype int8 --batch-size 56 --int8-bf16-mixed --ipex --jit --tasks lambada_openai \
                                    2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")

        if mode.endswith('woq8acc'):
            for model_id in data['modelargs'][mode]['modelid']:
                for rank in data['modelargs'][mode]['localrank']:
                    lines.append(f"export local_rank={rank}")
                    lines.append("deepspeed_core_config ${local_rank}")
                    lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")
                    for dtype in data['modelargs'][mode]['dtype']:
                        if 'codegen' in model_id:                             
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --accuracy-only -m {model_id} --dtype int8 --int8-bf16-mixed --ipex --jit --tasks hellaswag --batch-size 56 \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        elif 'neox' in model_id:
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --accuracy-only -m {model_id} --dtype int8 --ipex --jit --tasks lambada_openai --batch-size 56 \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        elif 'falcon' in model_id:
                            lines.append(f"python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --accuracy-only -m {model_id} --dtype int8 --int8-bf16-mixed --ipex --jit --tasks lambada_openai --config-file utils/model_config/tiiuae_falcon-40b_config.json --batch-size 56 \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        else:
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --accuracy-only -m {model_id} --dtype int8 --int8-bf16-mixed --ipex --jit --tasks lambada_openai --batch-size 56 \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")

        if mode.endswith('bf16dsacc'):
            # lines.append("cd ./distributed")
            lines.append("unset KMP_AFFINITY")
            for model_id in data['modelargs'][mode]['modelid']:
                if 'falcon' in model_id: 
                    lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --ipex --jit --tasks lambada_openai --accuracy-only --config-file utils/model_config/tiiuae_falcon-40b_config.json --batch-size 56 \
                                    2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_ds-bfloat16_{data['launcher']['hw']}.log")                
                elif 'codegen' in model_id:
                    lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --ipex --jit --tasks hellaswag --accuracy-only --batch-size 56 \
                                    2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_ds-bfloat16_{data['launcher']['hw']}.log")                    
                elif 'gpt-j' in model_id:
                    lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --ipex --jit --tasks lambada_openai --accuracy-only --batch-size 56 \
                                    2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_ds-bfloat16_{data['launcher']['hw']}.log")                    
                else:
                    lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --ipex --jit --tasks lambada_openai --accuracy-only --batch-size 1 \
                                    2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_ds-bfloat16_{data['launcher']['hw']}.log")

        if mode.endswith('woqdsacc'):

            lines.append("unset KMP_AFFINITY")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    if 'int8' in dtype:
                        if 'neox' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --dtype float32 --ipex --jit --tasks lambada_openai --accuracy-only --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        elif 'falcon' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --int8-bf16-mixed --ipex --jit --tasks lambada_openai --accuracy-only --ipex-weight-only-quantization --config-file utils/model_config/tiiuae_falcon-40b_config.json --batch-size 56 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        elif 'codegen' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --int8-bf16-mixed --ipex --jit --tasks hellaswag --accuracy-only --ipex-weight-only-quantization --batch-size 56 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        else:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --int8-bf16-mixed --ipex --jit --tasks lambada_openai --accuracy-only --ipex-weight-only-quantization --batch-size 56 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                    elif 'int4' in dtype:
                        if 'neox' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --dtype float32 --ipex --jit --tasks lambada_openai --accuracy-only --ipex-weight-only-quantization --batch-size 56 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        elif 'falcon' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --int8-bf16-mixed --ipex --jit --tasks lambada_openai --accuracy-only --ipex-weight-only-quantization --batch-size 56 --config-file utils/model_config/tiiuae_falcon-40b_config.json\
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        elif 'codegen' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --int8-bf16-mixed --ipex --jit --tasks hellaswag --accuracy-only --ipex-weight-only-quantization --batch-size 56 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        else:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --int8-bf16-mixed --ipex --jit --tasks lambada_openai --accuracy-only --ipex-weight-only-quantization --batch-size 56 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")

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
    if args.hbm_weekly:
        yml_file = 'bench_hbm_weekly_docker.yml'
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
