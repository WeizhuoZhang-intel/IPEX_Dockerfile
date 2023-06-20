# Generate runnable scripts for benchmarking
import argparse
import yaml
parser = argparse.ArgumentParser("Generation script", add_help=False)
parser.add_argument("-k","--extra_kmp",action="store_true",default=False,help="llm extra kmp configuration")
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
    printf $1 |tee -a ${log_dir}/summary.log
    printf " ${latency[1]},${first_latency},${avg_latency},${p90_latency},${p99_latency} \\n" |tee -a ${log_dir}/summary.log
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
        if mode == "default":
            lines.append("# Run workload")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device {data['modelargs'][mode]['device'][0]} -m {model_id} --input-tokens {input_token} --dtype {dtype} --ipex --jit --token-latency 2>&1 | tee -a $log_dir/llm_{mode}_{model_id.replace('/','_')}_{dtype}_{input_token}.log")
                        lines.append(f"collect_perf_logs_llm llm_{mode}_{model_id.replace('/','_')}_{dtype}_{input_token}.log")
        if mode.endswith('int8'):
            if mode.startswith('gptj'):
                lines.append("# GPT-J quantization")
                lines.append(f"python {data['modelargs'][mode]['scriptname']} --quantize --inc_smooth_quant --lambada --output_dir {data['modelargs'][mode]['outputdir']} --jit --int8")
            if mode.startswith('llama'):
                lines.append("# LLaMA quantization")
                lines.append(f"python python {data['modelargs'][mode]['scriptname']} --ipex_smooth_quant --lambada --output_dir {data['modelargs'][mode]['outputdir']} --jit --int8")
            lines.append("# Run workload")
            for input_token in data['modelargs'][mode]['inputtokens']:
                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --quantized_model_path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --benchmark --jit --int8 --token-latency 2>&1 | tee -a $log_dir/llm_{mode}_{input_token}.log")
                lines.append(f"collect_perf_logs_llm llm_{mode}_{input_token}.log")        
        if mode == 'deepspeed':
            lines.append("# DS Env config")
            lines.append(f"export OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']}")
            lines.append("unset KMP_AFFINITY")
            lines.append("# Run workload")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        lines.append(f"deepspeed --bind_cores_to_rank python {data['modelargs'][mode]['scriptname']} --benchmark --device {data['modelargs'][mode]['device'][0]} -m {model_id} --dtype {dtype} --input-tokens {input_token} --ipex --jit --token-latency 2>&1 | tee -a $log_dir/llm_{mode}_{model_id.replace('/','_')}_{dtype}_{input_token}.log") 
                        lines.append(f"collect_perf_logs_llm llm_{mode}_{model_id.replace('/','_')}_{dtype}_{input_token}.log")                        
        lines.append("")
        runfile.writelines([line + "\n" for line in lines])
    return generated_file


if __name__ == '__main__':
    for mode in 'default','gptj_int8','llama_int8','deepspeed':
        generate_commands('bench_preci.yml',mode,args.extra_kmp)
