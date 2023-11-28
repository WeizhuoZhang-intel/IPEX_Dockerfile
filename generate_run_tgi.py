# Generate runnable scripts for benchmarking
import argparse
import yaml
parser = argparse.ArgumentParser("Generation script", add_help=False)
parser.add_argument("--debug",action="store_true",default=False,help="only for debug regular track")
args = parser.parse_args()


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

def generate_commands(yml_file,mode):
    data = yaml.load(open(yml_file, 'r'),Loader=yaml.FullLoader)
    generated_file = "run_"+mode+".sh"
    with open(generated_file, "w") as runfile:
        lines = []
        lines.append("#!/bin/bash")

        lines.append("# Env config")
        lines.append(f"export repopath={data['envconfig']['REPOPATH']}")
        lines.append("while true")
        lines.append("do")
        lines.append(f'if [ `grep -c "Invalid hostname, defaulting to 0.0.0.0" $repopath/data/serve.log` -ne \'0\' ];then')
        lines.append("set -x")
        if mode.endswith('bf16'):
            for output_token in data['modelargs'][mode]['maxnewtokens']:
                for num in range(data['envconfig']['ITER']):

                    filename = str(output_token) + "file"+str(num)+".log"
                    lines.append(f"export filen={filename}")
                    lines.append(f"curl_cmd=\'curl {data['envconfig']['TRUEIP']}:8088/generate -X POST -d \'{{\"inputs\":\"{data['envconfig']['INPUT']}\", \"parameters\":{{\"max_new_tokens\":{output_token},\"do_sample\":{data['modelargs'][mode]['sample']} }} }}\' -H \'Content-Type: application/json\'\'")
                    lines.append(f"echo \"$curl_cmd\" | tee -a $repopath/data/$filen")
                    lines.append(f"eval \"$curl_cmd\" | tee -a $repopath/data/$filen")
                    lines.append("sleep 2s")


        lines.append("fi")
        lines.append("done")
        lines.append("")
        runfile.writelines([line + "\n" for line in lines])
    return generated_file


if __name__ == '__main__':
    yml_file = 'bench_preci.yml'
    if args.debug:
        yml_file = 'bench_tgi_debug.yml'
    data = yaml.load(open(yml_file, 'r'),Loader=yaml.FullLoader) 
    for mode in data['modelargs'].keys():
        generate_commands(yml_file, mode)


# awk '/time_per_token/ {gsub(/"/, "", $43); print $43}' text-generation-inference/data/serve.log | cut -d= -f2 
# 扣time_per_token的值
# awk '/time_per_token/ {gsub(/"/, "", $43); print $43}' text-generation-inference/data/serve.log
# 扣time_per_token整体