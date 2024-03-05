# Generate runnable scripts for benchmarking
import argparse
import yaml
parser = argparse.ArgumentParser("Generation script", add_help=False)
parser.add_argument("-k","--extra_kmp",action="store_true",default=False,help="llm extra kmp configuration")
parser.add_argument("-d","--deepspeed",action="store_true",default=False,help="only for deepspeed")
parser.add_argument("-y", "--yml_file", default="", help="which yml to use")
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
        lines.append("# Install Memory Profiler")
        lines.append(f"pip install -U memory_profiler")
        lines.append("export HF_HOME=/root/.cache/huggingface")
        lines.append("export TRANSFORMERS_OFFLINE=0")
        lines.append("bash token.sh")
        lines.append("# Env config")
        lines.append(f"export LD_PRELOAD={data['envconfig']['LD_PRELOAD']}")
        lines.append(f"export KMP_BLOCKTIME={data['envconfig']['KMP_BLOCKTIME']}")
        lines.append(f"export KMP_AFFINITY={data['envconfig']['KMP_AFFINITY']}")
        if extra_kmp:
            lines.append(f"export KMP_TPAUSE={data['envconfig']['LLM_EXTRA_KMP']['KMP_TPAUSE']}")
            lines.append(f"export KMP_SETTINGS={data['envconfig']['LLM_EXTRA_KMP']['KMP_SETTINGS']}")
            lines.append(f"export KMP_FORJOIN_BARRIER_PATTERN={data['envconfig']['LLM_EXTRA_KMP']['KMP_FORJOIN_BARRIER_PATTERN']}")
            lines.append(f"export KMP_PLAIN_BARRIER_PATTERN={data['envconfig']['LLM_EXTRA_KMP']['KMP_PLAIN_BARRIER_PATTERN']}")
            lines.append(f"export KMP_REDUCTION_BARRIER_PATTERN={data['envconfig']['LLM_EXTRA_KMP']['KMP_REDUCTION_BARRIER_PATTERN']}")
        lines.append("log_dir=${1:-log_dir}")
        lines.append("mkdir -p $log_dir")
        lines.append("# device info")
        lines.append(fetch_device_info)
        lines.append(collect_result)   
        
        lines.append("")
        if mode.startswith('default'):
            lines.append("# Run workload")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            for beam in data['modelargs'][mode]['greedy']:
                                for bs in data['modelargs'][mode]['batchsize']:
                                    for rank in data['modelargs'][mode]['localrank']:
                                        lines.append(f"mprof clean")
                                        lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} mprof run python {data['modelargs'][mode]['scriptname']} -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --batch-size {bs} --dtype {dtype} --disable-deployment-mode --profile --benchmark --num-iter {data['launcher']['iternum']} --num-warmup 5 2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                        lines.append("ut_result=${PIPESTATUS[0]}")
                                        lines.append(f"collect_perf_logs_llm $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log $ut_result")
                                        # lines.append(f"mv mprofile_*.dat $log_dir/llm_{mode}_{model_id.replace('/','_')}_{dtype}_{input_token}_mprofile.dat")
        elif mode.startswith('deepspeed'):
            lines.append("# DS Env config")
            lines.append(f"export OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']}")
            lines.append("unset KMP_AFFINITY")
            lines.append("# Run workload")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            for beam in data['modelargs'][mode]['greedy']:
                                for bs in data['modelargs'][mode]['batchsize']:
                                    for rank in data['modelargs'][mode]['localrank']:  
                                        lines.append(f"mprof clean")
                                        lines.append(f"export local_rank={rank}")
                                        lines.append("deepspeed_core_config ${local_rank}")
                                        lines.append("export CCL_WORKER_AFFINITY=${deepspeed_cores_list}")
                                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")

                                        lines.append("export FI_PROVIDER=tcp")

                                        if 'falcon' in model_id:
                                            lines.append(f"timeout 30m mprof run deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list {data['modelargs'][mode]['scriptname']} --benchmark -m {model_id} --dtype bfloat16 --batch-size {bs} --input-tokens {input_token} --ipex --token-latency --num-iter {data['launcher']['iternum']} --profile --num-warmup 5 --autotp --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                          2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log") 
                                        elif "mpt" in model_id:
                                            lines.append(f"timeout 30m mprof run deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list {data['modelargs'][mode]['scriptname']} --benchmark -m {model_id} --dtype bfloat16 --batch-size {bs} --input-tokens {input_token} --ipex --token-latency --num-iter {data['launcher']['iternum']} --profile --num-warmup 5 --autotp --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                          2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log") 
                                        else:
                                            lines.append(f"timeout 30m mprof run deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list {data['modelargs'][mode]['scriptname']} --benchmark -m {model_id} --dtype bfloat16 --batch-size {bs} --input-tokens {input_token} --ipex --token-latency --num-iter {data['launcher']['iternum']} --profile --num-warmup 5 --autotp 2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log") 
                                        lines.append("ut_result=${PIPESTATUS[0]}")
                                        lines.append(f"collect_perf_logs_llm llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log $ut_result")
                                        lines.append(f"mv mprofile_*.dat $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}_mprofile.dat")
        
        elif mode.startswith('stockds'):
            lines.append("# DS Env config")
            lines.append(f"export OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']}")
            lines.append("unset KMP_AFFINITY")
            lines.append("# Run workload")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            for beam in data['modelargs'][mode]['greedy']:
                                for bs in data['modelargs'][mode]['batchsize']:
                                    for rank in data['modelargs'][mode]['localrank']:  
                                        lines.append(f"mprof clean")
                                        lines.append(f"export local_rank={rank}")
                                        lines.append("deepspeed_core_config ${local_rank}")
                                        lines.append("export CCL_WORKER_AFFINITY=${deepspeed_cores_list}")
                                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")

                                        lines.append("export FI_PROVIDER=tcp")
                                        lines.append(f"timeout 30m mprof run deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list {data['modelargs'][mode]['scriptname']} --benchmark -m {model_id} --dtype bfloat16 --batch-size {bs} --input-tokens {input_token} --disable-deployment-mode --num-iter {data['launcher']['iternum']}  --profile --num-warmup 5 --autotp 2>&1 | tee -a $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log") 
                                        lines.append("ut_result=${PIPESTATUS[0]}")
                                        lines.append(f"collect_perf_logs_llm llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log $ut_result")
                                        lines.append(f"mv mprofile_*.dat $log_dir/llm_default_{(model_id.replace('/','-')).replace('_','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}_mprofile.dat")
        
        
        
        
        
        lines.append("")
        runfile.writelines([line + "\n" for line in lines])
    return generated_file


if __name__ == '__main__':
    #for mode in 'default','gptj_int8','llama_int8','deepspeed':
    if args.yml_file == "":
        yml_file = 'bench_preci.yml'
        if args.deepspeed:
            yml_file = 'bench_ds_preci.yml'
    else:
        yml_file = args.yml_file
    data = yaml.load(open(yml_file, 'r'),Loader=yaml.FullLoader) 
    for mode in data['modelargs'].keys():
        generate_commands(yml_file, mode, args.extra_kmp)
