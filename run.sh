# Env config
export KMP_BLOCKTIME=1
export KMP_AFFINITY="granularity=fine,compact,1,0"
# IOMP & TcMalloc
export LD_PRELOAD=/root/anaconda3/envs/llm/lib/libiomp5.so:/root/anaconda3/envs/llm/lib/libtcmalloc.so:${LD_PRELOAD}

# Run GPT-j workload
numactl -N 0 -m 0 python run_gptj.py --ipex --jit --dtype bfloat16 --max-new-tokens 32
