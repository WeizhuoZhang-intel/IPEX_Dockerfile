# Generate runnable scripts for benchmarking
import argparse
import yaml
parser = argparse.ArgumentParser("Generation script", add_help=False)
parser.add_argument("-k","--extra_kmp",action="store_true",default=False,help="llm extra kmp configuration")
parser.add_argument("-d","--deepspeed",action="store_true",default=False,help="only for deepspeed")
parser.add_argument("--nightly",action="store_true",default=False,help="only for nightly regular track")
parser.add_argument("--weekly",action="store_true",default=False,help="only for weekly regular track")
parser.add_argument("--debug",action="store_true",default=False,help="only for weekly regular track")
parser.add_argument("--emr_nightly",action="store_true",default=False,help="only for emr nightly regular track")
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
        if mode == 'deepspeed':
            lines.append("# DS Env config")
            lines.append(f"export OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']}")
            lines.append("unset KMP_AFFINITY")
            lines.append("# Run workload")
            lines.append("export CCL_WORKER_COUNT=1")
            lines.append("export CCL_PROCESS_LAUNCHER=none")
            lines.append("export CCL_ATL_TRANSPORT=ofi")
            lines.append("export CCL_ATL_SHM=1")
            lines.append("export CCL_WORKER_AFFINITY=0,32")
            lines.append("export DS_SHM_ALLREDUCE=1")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for beam in data['modelargs'][mode]['greedy']:
                            for numa in data['modelargs'][mode]['localrank']:
                                lines.append(f"nohup bash /root/workspace/get_mem.sh  >> $log_dir/mem-usage-llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}_NUMA_{numa}.log 2>&1 || true &")
                                if beam == True:
                                    lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {numa} --bind_core_list 0-63 {data['modelargs'][mode]['scriptname']} --benchmark --device {data['modelargs'][mode]['device'][0]} -m {model_id} --greedy --dtype {dtype} --input-tokens {input_token} \
                                    --ipex --jit --token-latency 2>&1 | tee -a $log_dir/llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}_NUMA_{numa}.log") 
                                else:
                                    lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {numa} --bind_core_list 0-63 {data['modelargs'][mode]['scriptname']} --benchmark --device {data['modelargs'][mode]['device'][0]} -m {model_id} --dtype {dtype} --input-tokens {input_token} \
                                    --ipex --jit --token-latency 2>&1 | tee -a $log_dir/llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}_NUMA_{numa}.log") 
                                lines.append(f"collect_perf_logs_llm llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}_NUMA_{numa}.log")                        
        if mode == 'deepspeed2S':
            lines.append("# DS Env config")
            lines.append(f"export OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']}")
            lines.append("unset KMP_AFFINITY")
            lines.append("# Run workload")
            lines.append("export CCL_WORKER_COUNT=1")
            lines.append("export CCL_PROCESS_LAUNCHER=none")
            lines.append("export CCL_ATL_TRANSPORT=ofi")
            lines.append("export CCL_ATL_SHM=1")
            lines.append("export CCL_WORKER_AFFINITY=0,32,64,96")
            lines.append("export DS_SHM_ALLREDUCE=1")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for beam in data['modelargs'][mode]['greedy']:
                            for numa in data['modelargs'][mode]['localrank']:
                                lines.append(f"nohup bash /root/workspace/get_mem.sh  >> $log_dir/mem-usage-llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}_NUMA_{numa}.log 2>&1 || true &")
                                if beam == True:
                                    lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {numa} --bind_core_list 0-127 {data['modelargs'][mode]['scriptname']} --benchmark --device {data['modelargs'][mode]['device'][0]} -m {model_id} --greedy --dtype {dtype} --input-tokens {input_token} \
                                    --ipex --jit --token-latency 2>&1 | tee -a $log_dir/llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}_NUMA_{numa}.log") 
                                else:
                                    lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {numa} --bind_core_list 0-127 {data['modelargs'][mode]['scriptname']} --benchmark --device {data['modelargs'][mode]['device'][0]} -m {model_id} --dtype {dtype} --input-tokens {input_token} \
                                    --ipex --jit --token-latency 2>&1 | tee -a $log_dir/llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}_NUMA_{numa}.log") 
                                lines.append(f"collect_perf_logs_llm llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}_NUMA_{numa}.log")     
        if mode == 'deepspeedneox2s':
            lines.append("# DS Env config")
            lines.append(f"export OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']}")
            lines.append("unset KMP_AFFINITY")
            lines.append("# Run workload")
            lines.append("export CCL_WORKER_COUNT=1")
            lines.append("export CCL_PROCESS_LAUNCHER=none")
            lines.append("export CCL_ATL_TRANSPORT=ofi")
            lines.append("export CCL_ATL_SHM=1")
            lines.append("export CCL_WORKER_AFFINITY=0,32,64,96")
            lines.append("export DS_SHM_ALLREDUCE=1")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for beam in data['modelargs'][mode]['greedy']:
                            for numa in data['modelargs'][mode]['localrank']:
                                lines.append(f"nohup bash /root/workspace/get_mem.sh  >> $log_dir/mem-usage-llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}_NUMA_{numa}.log 2>&1 || true &")
                                if beam == True:
                                    lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {numa} --bind_core_list 0-127 {data['modelargs'][mode]['scriptname']} --benchmark --device {data['modelargs'][mode]['device'][0]} -m {model_id} --greedy --dtype {dtype} --input-tokens {input_token} \
                                    --ipex --ipex-woq --jit --token-latency 2>&1 | tee -a $log_dir/llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}_NUMA_{numa}.log") 
                                else:
                                    lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {numa} --bind_core_list 0-127 {data['modelargs'][mode]['scriptname']} --benchmark --device {data['modelargs'][mode]['device'][0]} -m {model_id} --dtype {dtype} --input-tokens {input_token} \
                                    --ipex --ipex-woq --jit --token-latency 2>&1 | tee -a $log_dir/llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}_NUMA_{numa}.log") 
                                lines.append(f"collect_perf_logs_llm llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}_NUMA_{numa}.log")  
        if mode == 'deepspeedneox1s':
            lines.append("# DS Env config")
            lines.append(f"export OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']}")
            lines.append("unset KMP_AFFINITY")
            lines.append("# Run workload")
            lines.append("export CCL_WORKER_COUNT=1")
            lines.append("export CCL_PROCESS_LAUNCHER=none")
            lines.append("export CCL_ATL_TRANSPORT=ofi")
            lines.append("export CCL_ATL_SHM=1")
            lines.append("export CCL_WORKER_AFFINITY=0,32")
            lines.append("export DS_SHM_ALLREDUCE=1")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for beam in data['modelargs'][mode]['greedy']:
                            for numa in data['modelargs'][mode]['localrank']:
                                lines.append(f"nohup bash /root/workspace/get_mem.sh  >> $log_dir/mem-usage-llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}_NUMA_{numa}.log 2>&1 || true &")
                                if beam == True:
                                    lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {numa} --bind_core_list 0-67 {data['modelargs'][mode]['scriptname']} --benchmark --device {data['modelargs'][mode]['device'][0]} -m {model_id} --greedy --dtype {dtype} --input-tokens {input_token} \
                                    --ipex --ipex-woq --jit --token-latency 2>&1 | tee -a $log_dir/llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}_NUMA_{numa}.log") 
                                else:
                                    lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {numa} --bind_core_list 0-127 {data['modelargs'][mode]['scriptname']} --benchmark --device {data['modelargs'][mode]['device'][0]} -m {model_id} --dtype {dtype} --input-tokens {input_token} \
                                    --ipex --ipex-woq --jit --token-latency 2>&1 | tee -a $log_dir/llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}_NUMA_{numa}.log") 
                                lines.append(f"collect_perf_logs_llm llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}_NUMA_{numa}.log")  
        
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
    if args.emr_nightly:
        yml_file = 'bench_nightly_emr.yml'
    data = yaml.load(open(yml_file, 'r'),Loader=yaml.FullLoader) 
    for mode in data['modelargs'].keys():
        generate_commands(yml_file, mode, args.extra_kmp)
