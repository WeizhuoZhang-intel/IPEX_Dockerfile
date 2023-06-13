# Generate runnable scripts for benchmarking
import argparse
import yaml
parser = argparse.ArgumentParser("Generation script", add_help=False)
parser.add_argument("-k","--extra_kmp",action="store_true",default=False,help="llm extra kmp configuration")
args = parser.parse_args()

fetch_device_info = '''
lscpu
uname -a
free -h
numactl -H
sockets_num=$(lscpu |grep 'Socket(s):' |sed 's/[^0-9]//g')
cores_per_socket=$(lscpu |grep 'Core(s) per socket:' |sed 's/[^0-9]//g')
phsical_cores_num=$(echo |awk -v sockets_num=${sockets_num} -v cores_per_socket=${cores_per_socket} '{
    print sockets_num * cores_per_socket;
}')
numa_nodes_num=$(numactl -H |grep 'node [0-9]* cpus: [0-9].*' |wc -l)
threads_per_core=$(lscpu |grep 'Thread(s) per core:' |sed 's/[^0-9]//g')
cores_per_node=$(numactl -H |grep "node 0 cpus:" |sed 's/.*://' |awk -v tpc=$threads_per_core '{print int(NF / tpc)}')
if [ "${OOB_HBM_FLAT}" != "" ];then
    hbm_index=$numa_nodes_num
else
    hbm_index=0
fi
# cores to use
if [ "${cores_per_instance,,}" == "1s" ];then
    cores_per_instance=${cores_per_socket}
elif [ "${cores_per_instance,,}" == "1n" ];then
    cores_per_instance=${cores_per_node}
fi
# cpu array
if [ "${numa_nodes_use}" == "all" ];then
    numa_nodes_use_='1,$'
elif [ "${numa_nodes_use}" == "0" ];then
    numa_nodes_use_=1
else
    numa_nodes_use_=${numa_nodes_use}
fi
if [ "${device}" != "cuda" ];then
    device_array=($(numactl -H |grep "node [0-9]* cpus:" |sed "s/.*node//;s/cpus://" |sed -n "${numa_nodes_use_}p" |awk -v cpn=${cores_per_node} '{for(i=1;i<=cpn+1;i++) {printf(" %s ",$i)} printf("\n");}' |grep '[0-9]' |awk -v cpi=${cores_per_instance} -v cpn=${cores_per_node} -v cores=${OOB_TOTAL_CORES_USE} -v hi=${hbm_index} '{
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

    export OMP_NUM_THREADS=$(echo ${device_array[0]} |awk -F, '{printf("%d", NF)}')
else
    if [ $(nvidia-smi -L |grep 'MIG' |wc -l) -ne 0 ];then
        device_array=($(nvidia-smi -L |grep 'MIG' |sed 's/.*UUID: *//;s/).*//' |sed -n "${numa_nodes_use_}p"))
    else
        device_array=($(nvidia-smi -L |grep 'NVIDIA' |sed 's/.*UUID: *//;s/).*//' |sed -n "${numa_nodes_use_}p"))
    fi
export CUDA_VISIBLE_DEVICES=${device_array[0]}
fi
instance=${#device_array[@]}
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
        lines.append("# device info")
        lines.append(fetch_device_info)
        lines.append("")
        if mode == "default":
            lines.append("# Run workload")
            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device {data['modelargs'][mode]['device'][0]} -m {data['modelargs'][mode]['modelid'][0]} --dtype {data['modelargs'][mode]['dtype'][0]} --ipex --jit ")
        if mode.endswith('int8'):
            if mode.startswith('gptj'):
                lines.append("# GPT-J quantization")
                lines.append(f"python python {data['modelargs'][mode]['scriptname']} --quantize --inc_smooth_quant --lambada --output_dir {data['modelargs'][mode]['outputdir']} --jit --int8")
            if mode.startswith('llama'):
                lines.append("# LLaMA quantization")
                lines.append(f"python python {data['modelargs'][mode]['scriptname']} --ipex_smooth_quant --lambada --output_dir {data['modelargs'][mode]['outputdir']} --jit --int8")
            lines.append("# Run workload")
            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --quantized_model_path {data['modelargs'][mode]['quantizedmodelpath']} --benchmark --jit --int8")          
        if mode == 'deepspeed':
            lines.append("# DS Env config")
            lines.append(f"export OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']}")
            lines.append("unset KMP_AFFINITY")
            lines.append("# Run workload")
            lines.append(f"deepspeed --bind_cores_to_rank python {data['modelargs'][mode]['scriptname']} --benchmark --device {data['modelargs'][mode]['device'][0]} -m {data['modelargs'][mode]['modelid'][0]} --dtype {data['modelargs'][mode]['dtype'][0]} --ipex --jit") 
        lines.append("")
        runfile.writelines([line + "\n" for line in lines])
    return generated_file
if __name__ == '__main__':
    for mode in 'default','gptj_int8','llama_int8','deepspeed':
        generate_commands('bench.yml',mode,args.extra_kmp)
