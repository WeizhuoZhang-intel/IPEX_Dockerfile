## ---------------------------------------------------------------- env setup ----------------------------------------------------------------------------------
export http_proxy=http://proxy-dmz.intel.com:912
export https_proxy=http://proxy-dmz.intel.com:912

conda create -n llm python=3.10 -y

conda activate llm

export no_proxy=127.0.0.1,localhost,ubit-artifactory-sh.intel.com,mlpc.intel.com

# install PT
pip install http://mlpc.intel.com/downloads/cpu/ipex-2.2/stock_pt_whls/torch-2.2.0+cpu-cp310-cp310-linux_x86_64.whl

# install internal IPEX 2.2 for now, will be available in public soon
pip install http://mlpc.intel.com/downloads/LLM/cpu-device-whl/2024_ww033_llm/intel_extension_for_pytorch-2.2.0+gita891a23-cp310-cp310-linux_x86_64.whl

# install lm-eval for accuracy test
git clone https://github.com/EleutherAI/lm-evaluation-harness.git 
cd lm-evaluation-harness
git checkout cc9778fbe4fa1a709be2abed9deb6180fd40e7e2
python setup.py bdist_wheel
pip install dist/*.whl
cd ..

# install dependency
pip install transformers==4.35.2 cpuid accelerate datasets sentencepiece protobuf==3.20.3
conda install -y mkl
conda install cmake ninja mkl mkl-include -y
conda install -y gperftools -c conda-forge

# Get latest internal IPEX LLM BKC scripts (2.2), they will be available in public soon
git clone -b release/2.2 https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu

cd frameworks.ai.pytorch.ipex-cpu/examples/cpu/inference/python/llm

wget https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/prompt.json

cp -r prompt.json single_instance/

# Get smoothquant recipes for quantization (for "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "facebook/opt-1.3b")
wget http://mlpc.intel.com/downloads/increcipe/llama-2-7b_qconfig.json
wget http://mlpc.intel.com/downloads/increcipe/llama-2-13b_qconfig.json
wget http://mlpc.intel.com/downloads/increcipe/opt-1b3_qconfig.json

# Activate environment variables
source ./tools/env_activate.sh

##------------------------------------------------------------------------------------- run cmd --------------------------------------------------------------------------------------------------
# The following "OMP_NUM_THREADS" and "numactl" settings are based on the assumption that
# the target server has 56 physical cores per numa socket, and we benchmark with 1 socket.
# Please adjust the settings per your hardware.

# ---------------------------------------------------------------------------------- FOR GENERATION ---------------------------------------------------------------------------------------------- 
# static int8 with smoothquant
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m facebook/opt-1.3b --ipex-smooth-quant --qconfig-summary-file opt-1b3_qconfig.json --output-dir "saved_results" --input-tokens 32 --max-new-tokens 32 --token-latency

OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-hf --ipex-smooth-quant --qconfig-summary-file llama-2-7b_qconfig.json --output-dir "saved_results" --input-tokens 32 --max-new-tokens 32 --token-latency

OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-13b-hf --ipex-smooth-quant --qconfig-summary-file llama-2-13b_qconfig.json --output-dir "saved_results" --input-tokens 32 --max-new-tokens 32 --token-latency

# ----------------------------------------------------------------------------------- FOR ACCURACY ------------------------------------------------------------------------------------------------

# static int8 with smoothquant
numactl -m 0 -C 0-55 python single_instance/run_accuracy.py --quantized-model-path saved_results/best_model.pt -m facebook/opt-1.3b --dtype int8 --ipex --tasks lambada_openai

numactl -m 0 -C 0-55 python single_instance/run_accuracy.py --quantized-model-path saved_results/best_model.pt -m meta-llama/Llama-2-7b-hf --dtype int8 --ipex --tasks lambada_openai

numactl -m 0 -C 0-55 python single_instance/run_accuracy.py --quantized-model-path saved_results/best_model.pt -m meta-llama/Llama-2-13b-hf --dtype int8 --ipex --tasks lambada_openai
# fp32
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python single_instance/run_accuracy.py -m facebook/opt-1.3b --dtype float32 --ipex --tasks lambada_openai

OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python single_instance/run_accuracy.py -m meta-llama/Llama-2-7b-hf --dtype float32 --ipex --tasks lambada_openai

OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python single_instance/run_accuracy.py -m meta-llama/Llama-2-13b-hf --dtype float32 --ipex --tasks lambada_openai