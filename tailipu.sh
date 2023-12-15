###---------------------------------------conda env creation ----###
conda create -n llm python=3.10 -y
conda activate llm

## install PT
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cpu

# internal IPEX 2.1.100 for now, will be available in public soon (if you can not get access on linux host, please download it from website via link: http://mlpc.intel.com/downloads/cpu/ipex-2.1.100/rc2/ )
pip install http://mlpc.intel.com/downloads/cpu/ipex-2.1.100/rc2/intel_extension_for_pytorch-2.1.100+cpu-cp310-cp310-linux_x86_64.whl

## install dependency
pip install transformers==4.31.0 cpuid accelerate datasets sentencepiece protobuf==3.20.3
conda install cmake ninja mkl mkl-include -y
conda install gperftools -c conda-forge -y

## install lm-eval for accuracy test
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness/
git checkout cc9778fbe4fa1a709be2abed9deb6180fd40e7e2
pip install -e .
cd ..

## Get latest internal IPEX LLM BKC scripts (2.1.100), they will be available in public soon
git clone https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu -b release/2.1
cd frameworks.ai.pytorch.ipex-cpu/examples/cpu/inference/python/llm

wget https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/prompt.json
cp prompt.json ./single_instance
cp prompt.json ./distributed
## Get smoothquant recipes for quantization (for "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf")
wget https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/llama-2-7b_qconfig.json
wget https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/llama-2-13b_qconfig.json

# Activate environment variables
source ./tools/env_activate.sh

###---------------------------------------Below is CMDs to run perf and acc tests on "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf" ----###

# The following "OMP_NUM_THREADS" and "numactl" settings are based on the assumption that
# the target server has 56 physical cores per numa socket, and we benchmark with 1 socket.
# Please adjust the settings per your hardware.


#### "meta-llama/Llama-2-7b-hf"

## Performance test CMD for "meta-llama/Llama-2-7b-hf"

# Running FP32  with IPEX
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype float32 --ipex --deployment-mode
# Running BF16  with IPEX
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex --deployment-mode
# INT8 static quantization
mkdir "saved_results_llama_7b_sq"
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-hf --ipex-smooth-quant --qconfig-summary-file "./llama-2-7b_qconfig.json" --output-dir "./saved_results_llama_7b_sq" --int8-bf16-mixed
# INT8 weight-only quantization
mkdir "saved_results_llama_7b_woq"
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-hf --ipex-weight-only-quantization  --output-dir "./saved_results_llama_7b_woq" --int8-bf16-mixed

## Accuracy test CMD for "meta-llama/Llama-2-7b-hf"

# Running FP32 
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python ./single_instance/run_accuracy.py --accuracy-only -m meta-llama/Llama-2-7b-hf --dtype float32 --ipex --jit --tasks lambada_openai
# Running BF16 
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python ./single_instance/run_accuracy.py --accuracy-only -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex --jit --tasks lambada_openai
# INT8 static quantization
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python ./single_instance/run_accuracy.py -m meta-llama/Llama-2-7b-hf --quantized-model-path "./saved_results_llama_7b_sq/best_model.pt" --dtype int8 --accuracy-only --jit --tasks lambada_openai --int8-bf16-mixed
# INT8 weight-only quantization
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python ./single_instance/run_accuracy.py -m meta-llama/Llama-2-7b-hf --quantized-model-path "./saved_results_llama_7b_woq/best_model.pt" --dtype int8 --accuracy-only --jit --tasks lambada_openai --int8-bf16-mixed


#### "meta-llama/Llama-2-13b-hf"

## Performance test CMD for "meta-llama/Llama-2-13b-hf"

# Running FP32  with IPEX
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-13b-hf --dtype float32 --ipex --deployment-mode
# Running BF16  with IPEX
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-13b-hf  --dtype bfloat16 --ipex --deployment-mode
# INT8 static quantization
mkdir "saved_results_llama_13b_sq"
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-13b-hf --ipex-smooth-quant --qconfig-summary-file "./llama-2-13b_qconfig.json" --output-dir "./saved_results_llama_13b_sq" --int8-bf16-mixed
# INT8 weight-only quantization
mkdir "saved_results_llama_13b_woq"
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-13b-hf  --ipex-weight-only-quantization  --output-dir "./saved_results_llama_13b_woq" --int8-bf16-mixed

## Accuracy test CMD for "meta-llama/Llama-2-13b-hf"

# Running FP32 
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python ./single_instance/run_accuracy.py --accuracy-only -m meta-llama/Llama-2-13b-hf  --dtype float32 --ipex --jit --tasks lambada_openai
# Running BF16 
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python ./single_instance/run_accuracy.py --accuracy-only -m meta-llama/Llama-2-13b-hf  --dtype bfloat16 --ipex --jit --tasks lambada_openai
# INT8 static quantization
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python ./single_instance/run_accuracy.py -m meta-llama/Llama-2-13b-hf --quantized-model-path "./saved_results_llama_13b_sq/best_model.pt" --dtype int8 --accuracy-only --jit --tasks lambada_openai --int8-bf16-mixed
# INT8 weight-only quantization
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python ./single_instance/run_accuracy.py -m meta-llama/Llama-2-13b-hf  --quantized-model-path "./saved_results_llama_13b_woq/best_model.pt" --dtype int8 --accuracy-only --jit --tasks lambada_openai --int8-bf16-mixed