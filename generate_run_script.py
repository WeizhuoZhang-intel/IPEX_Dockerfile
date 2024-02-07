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
 
collect_result = '''
function collect_perf_logs_llm() {
    ut_result=$2
    peak_mem=$(mprof peak | grep mprofile | awk '{print $2}')
    quant_peak_mem=${3-0}
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
    printf " ${latency[1]},${first_latency},${avg_latency},${p90_latency},${p99_latency},${ut_result},${peak_mem},${quant_peak_mem} \\n" |tee -a ${log_dir}/summary.log
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
        lines.append("pip install --upgrade huggingface_hub")
        lines.append("huggingface-cli login --token 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")
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

        lines.append("pip install --upgrade huggingface_hub")
        lines.append("huggingface-cli login --token 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")
        
        lines.append("")
        if mode.startswith('default'):
            lines.append("# Run workload")
            lines.append("huggingface-cli login --token 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")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        lines.append(f"mprof clean")
                        lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} mprof run python {data['modelargs'][mode]['scriptname']} -m {model_id} --input-tokens {input_token} --dtype {dtype} --disable-deployment-mode --profile --token-latency --benchmark --num-iter 15 2>&1 | tee -a $log_dir/llm_{mode}_{model_id.replace('/','_')}_{dtype}_{input_token}.log")
                        lines.append("ut_result=${PIPESTATUS[0]}")
                        lines.append(f"collect_perf_logs_llm llm_{mode}_{model_id.replace('/','_')}_{dtype}_{input_token}.log $ut_result")
                        lines.append(f"mv mprofile_*.dat $log_dir/llm_{mode}_{model_id.replace('/','_')}_{dtype}_{input_token}_mprofile.dat")
        elif mode.endswith('int8'):
            # int8 test for GPT-J and LLaMA
            lines.append(f"mprof clean")
            # if mode.startswith('gptj'):
            #     lines.append("# GPT-J quantization")
            #     lines.append(f"mprof run python run.py -m EleutherAI/gpt-j-6b --ipex-smooth-quant --batch-size 56 --calib-iters 100 --calib-shuffle --fallback-add --alpha 0.85 ")
            # if mode.startswith('llama'):
            #     lines.append("# LLaMA quantization")
            #     lines.append(f"mprof run python run.py -m meta-llama/Llama-2-7b-hf --ipex-smooth-quant --lambada --output_dir {data['modelargs'][mode]['outputdir']} --jit --int8")
            # lines.append("quant_peak_mem=$(mprof peak | grep mprofile | awk '{print $2}')")
            # lines.append("# Run workload")
            # lines.append(f"mprof clean")
            # for input_token in data['modelargs'][mode]['inputtokens']:
            #     lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} mprof run python {data['modelargs'][mode]['scriptname']} --input-tokens {input_token} --benchmark --jit --int8-bf16-mixed --token-latency --num-iter 20 2>&1 | tee -a $log_dir/llm_{mode}_{input_token}.log")
            #     lines.append("ut_result=${PIPESTATUS[0]}")
            #     lines.append(f"collect_perf_logs_llm llm_{mode}_{input_token}.log $ut_result $quant_peak_mem")
            #     lines.append(f"mv mprofile_*.dat $log_dir/llm_{mode}_{input_token}.dat")
        elif mode.startswith('deepspeed'):
            lines.append("# DS Env config")
            lines.append("huggingface-cli login --token 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")
            lines.append(f"export OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']}")
            lines.append("unset KMP_AFFINITY")
            lines.append("# Run workload")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        lines.append(f"mprof clean")
                        lines.append("export FI_PROVIDER=tcp")
                        lines.append(f"mprof run deepspeed --bind_cores_to_rank {data['modelargs'][mode]['scriptname']} --benchmark -m {model_id} --dtype {dtype} --input-tokens {input_token} --disable-deployment-mode --token-latency --num-iter 15 --profile  --autotp 2>&1 | tee -a $log_dir/llm_{mode}_{model_id.replace('/','_')}_{dtype}_{input_token}.log") 
                        lines.append("ut_result=${PIPESTATUS[0]}")
                        lines.append(f"collect_perf_logs_llm llm_{mode}_{model_id.replace('/','_')}_{dtype}_{input_token}.log $ut_result")
                        lines.append(f"mv mprofile_*.dat $log_dir/llm_{mode}_{model_id.replace('/','_')}_{dtype}_{input_token}_mprofile.dat")
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
