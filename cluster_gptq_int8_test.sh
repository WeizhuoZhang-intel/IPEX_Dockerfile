#!/bin/bash
# set -x

set -x
pwd

export log_dir=/mnt/aitrgdata/mint/ipex24rc0/logs
export upath=/mnt/aitrgdata/mint/ipex24rc0/gptq

mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/baichuan-inc/Baichuan-13B-Chat
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/baichuan-inc/Baichuan-13B-Chat --quant-with-amp --lowp-mode INT8 -m baichuan-inc/Baichuan-13B-Chat --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/baichuan-inc/Baichuan-13B-Chat/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/baichuan-inc/Baichuan-13B-Chat/best_model.pt -m baichuan-inc/Baichuan-13B-Chat --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/Baichuan-13B-Chat-woq-int4-int8-acc.log

mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/baichuan-inc/Baichuan2-13B-Chat
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/baichuan-inc/Baichuan2-13B-Chat --quant-with-amp --lowp-mode INT8 -m baichuan-inc/Baichuan2-13B-Chat --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/baichuan-inc/Baichuan2-13B-Chat/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/baichuan-inc/Baichuan2-13B-Chat/best_model.pt -m baichuan-inc/Baichuan2-13B-Chat --dtype int8 --ipex --quant-with-amp --tasks hellaswag 2>&1 | tee -a $log_dir/Baichuan2-13B-Chat-woq-int4-int8-acc.log

mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/baichuan-inc/Baichuan2-7B-Chat
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/baichuan-inc/Baichuan2-7B-Chat --quant-with-amp --lowp-mode INT8 -m baichuan-inc/Baichuan2-7B-Chat --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/baichuan-inc/Baichuan2-7B-Chat/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/baichuan-inc/Baichuan2-7B-Chat/best_model.pt -m baichuan-inc/Baichuan2-7B-Chat --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/Baichuan2-7B-Chat-woq-int4-int8-acc.log

mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/bigcode/starcoder
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/bigcode/starcoder --quant-with-amp --lowp-mode INT8 -m bigcode/starcoder --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/bigcode/starcoder/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/bigcode/starcoder/best_model.pt -m bigcode/starcoder --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/starcoder-woq-int4-int8-acc.log

mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/bigscience/bloom-1b7
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/bigscience/bloom-1b7 --quant-with-amp --lowp-mode INT8 -m bigscience/bloom-1b7 --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/bigscience/bloom-1b7/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/bigscience/bloom-1b7/best_model.pt -m bigscience/bloom-1b7 --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/bloom-woq-int4-int8-acc.log

mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/databricks/dolly-v2-12b
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/databricks/dolly-v2-12b --quant-with-amp --lowp-mode INT8 -m databricks/dolly-v2-12b --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/databricks/dolly-v2-12b/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/databricks/dolly-v2-12b/best_model.pt -m databricks/dolly-v2-12b --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/dolly-woq-int4-int8-acc.log

mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/EleutherAI/gpt-j-6b
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/EleutherAI/gpt-j-6b --quant-with-amp --lowp-mode INT8 -m EleutherAI/gpt-j-6b --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/EleutherAI/gpt-j-6b/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/EleutherAI/gpt-j-6b/best_model.pt -m EleutherAI/gpt-j-6b --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/gpt-j-woq-int4-int8-acc.log


mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/EleutherAI/gpt-neox-20b
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/EleutherAI/gpt-neox-20b --quant-with-amp --lowp-mode INT8 -m EleutherAI/gpt-neox-20b --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/EleutherAI/gpt-neox-20b/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/EleutherAI/gpt-neox-20b/best_model.pt -m EleutherAI/gpt-neox-20b --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/gpt-neox-woq-int4-int8-acc.log


mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/facebook/opt-1.3b
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/facebook/opt-1.3b --quant-with-amp --lowp-mode INT8 -m facebook/opt-1.3b --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/facebook/opt-1.3b/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/facebook/opt-1.3b/best_model.pt -m facebook/opt-1.3b --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/opt-1.3b-woq-int4-int8-acc.log

mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/facebook/opt-30b
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/facebook/opt-30b --quant-with-amp --lowp-mode INT8 -m facebook/opt-30b --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/facebook/opt-30b/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/facebook/opt-30b/best_model.pt -m facebook/opt-30b --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/opt-30b-woq-int4-int8-acc.log


#!/bin/bash
# set -x

set -x
pwd

export log_dir=/mnt/aitrgdata/mint/ipex24rc0/logs
export upath=/mnt/aitrgdata/mint/ipex24rc0/gptq


mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/google/flan-t5-xl
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/google/flan-t5-xl --quant-with-amp --lowp-mode INT8 -m google/flan-t5-xl --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/google/flan-t5-xl/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/google/flan-t5-xl/best_model.pt -m google/flan-t5-xl --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/flan-t5-woq-int4-int8-acc.log


mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/meta-llama/Llama-2-13b-hf
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/meta-llama/Llama-2-13b-hf --quant-with-amp --lowp-mode INT8 -m meta-llama/Llama-2-13b-hf --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/meta-llama/Llama-2-13b-hf/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/meta-llama/Llama-2-13b-hf/best_model.pt -m meta-llama/Llama-2-13b-hf --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/Llama-2-13b-hf-woq-int4-int8-acc.log


mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/meta-llama/Llama-2-70b-hf
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/meta-llama/Llama-2-70b-hf --quant-with-amp --lowp-mode INT8 -m meta-llama/Llama-2-70b-hf --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/meta-llama/Llama-2-70b-hf/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/meta-llama/Llama-2-70b-hf/best_model.pt -m meta-llama/Llama-2-70b-hf --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/Llama-2-70b-woq-int4-int8-acc.log


mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/meta-llama/Llama-2-7b-hf
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/meta-llama/Llama-2-7b-hf --quant-with-amp --lowp-mode INT8 -m meta-llama/Llama-2-7b-hf --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/meta-llama/Llama-2-7b-hf/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/meta-llama/Llama-2-7b-hf/best_model.pt -m meta-llama/Llama-2-7b-hf --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/Llama-2-7b-hf-woq-int4-int8-acc.log


mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/meta-llama/Meta-Llama-3-70B
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/meta-llama/Meta-Llama-3-70B --quant-with-amp --lowp-mode INT8 -m meta-llama/Meta-Llama-3-70B --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/meta-llama/Meta-Llama-3-70B/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/meta-llama/Meta-Llama-3-70B/best_model.pt -m meta-llama/Meta-Llama-3-70B --dtype int8 --ipex --quant-with-amp --tasks piqa 2>&1 | tee -a $log_dir/Meta-Llama-3-70B-woq-int4-int8-acc.log



mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/meta-llama/Meta-Llama-3-8B
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/meta-llama/Meta-Llama-3-8B --quant-with-amp --lowp-mode INT8 -m meta-llama/Meta-Llama-3-8B --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/meta-llama/Meta-Llama-3-8B/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/meta-llama/Meta-Llama-3-8B/best_model.pt -m meta-llama/Meta-Llama-3-8B --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/Meta-Llama-3-8B-woq-int4-int8-acc.log



mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/microsoft/phi-2
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/microsoft/phi-2 --quant-with-amp --lowp-mode INT8 -m microsoft/phi-2 --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/microsoft/phi-2/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/microsoft/phi-2/best_model.pt -m microsoft/phi-2 --dtype int8 --ipex --quant-with-amp --tasks hellaswag 2>&1 | tee -a $log_dir/phi-2-woq-int4-int8-acc.log


mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/microsoft/Phi-3-medium-128k-instruct
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/microsoft/Phi-3-medium-128k-instruct --quant-with-amp --lowp-mode INT8 -m microsoft/Phi-3-medium-128k-instruct --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/microsoft/Phi-3-medium-128k-instruct/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/microsoft/Phi-3-medium-128k-instruct/best_model.pt -m microsoft/Phi-3-medium-128k-instruct --dtype int8 --ipex --quant-with-amp --tasks hellaswag 2>&1 | tee -a $log_dir/Phi-3-medium-128k-woq-int4-int8-acc.log


mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/microsoft/Phi-3-medium-4k-instruct
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/microsoft/Phi-3-medium-4k-instruct --quant-with-amp --lowp-mode INT8 -m microsoft/Phi-3-medium-4k-instruct --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/microsoft/Phi-3-medium-4k-instruct/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/microsoft/Phi-3-medium-4k-instruct/best_model.pt -m microsoft/Phi-3-medium-4k-instruct --dtype int8 --ipex --quant-with-amp --tasks hellaswag 2>&1 | tee -a $log_dir/Phi-3-medium-4k-woq-int4-int8-acc.log

#!/bin/bash
# set -x

set -x
pwd

export log_dir=/mnt/aitrgdata/mint/ipex24rc0/logs
export upath=/mnt/aitrgdata/mint/ipex24rc0/gptq


mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/microsoft/Phi-3-mini-128k-instruct
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/microsoft/Phi-3-mini-128k-instruct --quant-with-amp --lowp-mode INT8 -m microsoft/Phi-3-mini-128k-instruct --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/microsoft/Phi-3-mini-128k-instruct/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/microsoft/Phi-3-mini-128k-instruct/best_model.pt -m microsoft/Phi-3-mini-128k-instruct --dtype int8 --ipex --quant-with-amp --tasks hellaswag 2>&1 | tee -a $log_dir/Phi-3-mini-128k-woq-int4-int8-acc.log


mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/microsoft/Phi-3-mini-4k-instruct
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/microsoft/Phi-3-mini-4k-instruct --quant-with-amp --lowp-mode INT8 -m microsoft/Phi-3-mini-4k-instruct --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/microsoft/Phi-3-mini-4k-instruct/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/microsoft/Phi-3-mini-4k-instruct/best_model.pt -m microsoft/Phi-3-mini-4k-instruct --dtype int8 --ipex --quant-with-amp --tasks hellaswag 2>&1 | tee -a $log_dir/Phi-3-mini-4k-woq-int4-int8-acc.log



mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/mistralai/Mixtral-8x7B-v0.1
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/mistralai/Mixtral-8x7B-v0.1 --quant-with-amp --lowp-mode INT8 -m mistralai/Mixtral-8x7B-v0.1 --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/mistralai/Mixtral-8x7B-v0.1/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/mistralai/Mixtral-8x7B-v0.1/best_model.pt -m mistralai/Mixtral-8x7B-v0.1 --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/Mixtral-8x7B-v0.1-woq-int4-int8-acc.log


mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/mosaicml/mpt-7b
python single_instance/run_quantization.py --ipex-weight-only-quantization --config-file=utils/model_config/mosaicml_mpt-7b_config.json --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/mosaicml/mpt-7b --quant-with-amp --lowp-mode INT8 -m mosaicml/mpt-7b --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/mosaicml/mpt-7b/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/mosaicml/mpt-7b/best_model.pt --config-file=utils/model_config/mosaicml_mpt-7b_config.json -m mosaicml/mpt-7b --dtype int8 --ipex --quant-with-amp --tasks hellaswag 2>&1 | tee -a $log_dir/mpt-7b-woq-int4-int8-acc.log


mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/Qwen/Qwen2-7B
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/Qwen/Qwen2-7B --quant-with-amp --lowp-mode INT8 -m Qwen/Qwen2-7B --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/Qwen/Qwen2-7B/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/Qwen/Qwen2-7B/best_model.pt -m Qwen/Qwen2-7B --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/Qwen2-7B-woq-int4-int8-acc.log


mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/Qwen/Qwen-7B-Chat
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/Qwen/Qwen-7B-Chat --quant-with-amp --lowp-mode INT8 -m Qwen/Qwen-7B-Chat --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/Qwen/Qwen-7B-Chat/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/Qwen/Qwen-7B-Chat/best_model.pt -m Qwen/Qwen-7B-Chat --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/Qwen-7B-Chat-woq-int4-int8-acc.log


mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/Salesforce/codegen-2B-multi
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/Salesforce/codegen-2B-multi --quant-with-amp --lowp-mode INT8 -m Salesforce/codegen-2B-multi --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/Salesforce/codegen-2B-multi/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/Salesforce/codegen-2B-multi/best_model.pt -m Salesforce/codegen-2B-multi --dtype int8 --ipex --quant-with-amp --tasks hellaswag 2>&1 | tee -a $log_dir/codegen-woq-int4-int8-acc.log


mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/stabilityai/stablelm-2-1_6b
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/stabilityai/stablelm-2-1_6b --quant-with-amp --lowp-mode INT8 -m stabilityai/stablelm-2-1_6b --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/stabilityai/stablelm-2-1_6b/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/stabilityai/stablelm-2-1_6b/best_model.pt -m stabilityai/stablelm-2-1_6b --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/stablelm-woq-int4-int8-acc.log


mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/THUDM/chatglm2-6b
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/THUDM/chatglm2-6b --quant-with-amp --lowp-mode INT8 -m THUDM/chatglm2-6b --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/THUDM/chatglm2-6b/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/THUDM/chatglm2-6b/best_model.pt -m THUDM/chatglm2-6b --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/chatglm2-6b-woq-int4-int8-acc.log


mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/THUDM/chatglm3-6b
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/THUDM/chatglm3-6b --quant-with-amp --lowp-mode INT8 -m THUDM/chatglm3-6b --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/THUDM/chatglm3-6b/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/THUDM/chatglm3-6b/best_model.pt -m THUDM/chatglm3-6b --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/chatglm3-6b-woq-int4-int8-acc.log


mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/tiiuae/falcon-40b
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/tiiuae/falcon-40b --quant-with-amp --lowp-mode INT8 -m tiiuae/falcon-40b --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/tiiuae/falcon-40b/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/tiiuae/falcon-40b/best_model.pt -m tiiuae/falcon-40b --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/falcon-40b-woq-int4-int8-acc.log


mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/tiiuae/falcon-7b
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/tiiuae/falcon-7b --quant-with-amp --lowp-mode INT8 -m tiiuae/falcon-7b --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/tiiuae/falcon-7b/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/tiiuae/falcon-7b/best_model.pt -m tiiuae/falcon-7b --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/falcon-7b-woq-int4-int8-acc.log





















## ---------------------------------------------- addition ------------------------------------------------------


#!/bin/bash
# set -x

set -x
pwd

export log_dir=/mnt/aitrgdata/mint/ipex24rc0/logs
export upath=/mnt/aitrgdata/mint/ipex24rc0/gptq

mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/Qwen/Qwen-7B-Chat
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/Qwen/Qwen-7B-Chat --quant-with-amp --lowp-mode INT8 -m Qwen/Qwen-7B-Chat --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/Qwen/Qwen-7B-Chat/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/Qwen/Qwen-7B-Chat/best_model.pt -m Qwen/Qwen-7B-Chat --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/Qwen-7B-Chat-woq-int4-int8-acc.log


mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/THUDM/chatglm2-6b
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/THUDM/chatglm2-6b --quant-with-amp --lowp-mode INT8 -m THUDM/chatglm2-6b --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/THUDM/chatglm2-6b/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/THUDM/chatglm2-6b/best_model.pt -m THUDM/chatglm2-6b --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/chatglm2-6b-woq-int4-int8-acc.log


mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/THUDM/chatglm3-6b
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/THUDM/chatglm3-6b --quant-with-amp --lowp-mode INT8 -m THUDM/chatglm3-6b --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/THUDM/chatglm3-6b/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/THUDM/chatglm3-6b/best_model.pt -m THUDM/chatglm3-6b --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/chatglm3-6b-woq-int4-int8-acc.log

mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/baichuan-inc/Baichuan-13B-Chat
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/baichuan-inc/Baichuan-13B-Chat --quant-with-amp --lowp-mode INT8 -m baichuan-inc/Baichuan-13B-Chat --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/baichuan-inc/Baichuan-13B-Chat/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/baichuan-inc/Baichuan-13B-Chat/best_model.pt -m baichuan-inc/Baichuan-13B-Chat --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/Baichuan-13B-Chat-woq-int4-int8-acc.log

mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/baichuan-inc/Baichuan2-13B-Chat
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/baichuan-inc/Baichuan2-13B-Chat --quant-with-amp --lowp-mode INT8 -m baichuan-inc/Baichuan2-13B-Chat --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/baichuan-inc/Baichuan2-13B-Chat/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/baichuan-inc/Baichuan2-13B-Chat/best_model.pt -m baichuan-inc/Baichuan2-13B-Chat --dtype int8 --ipex --quant-with-amp --tasks hellaswag 2>&1 | tee -a $log_dir/Baichuan2-13B-Chat-woq-int4-int8-acc.log

mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/baichuan-inc/Baichuan2-7B-Chat
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/baichuan-inc/Baichuan2-7B-Chat --quant-with-amp --lowp-mode INT8 -m baichuan-inc/Baichuan2-7B-Chat --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/baichuan-inc/Baichuan2-7B-Chat/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/baichuan-inc/Baichuan2-7B-Chat/best_model.pt -m baichuan-inc/Baichuan2-7B-Chat --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/Baichuan2-7B-Chat-woq-int4-int8-acc.log

mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/bigcode/starcoder
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/bigcode/starcoder --quant-with-amp --lowp-mode INT8 -m bigcode/starcoder --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/bigcode/starcoder/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/bigcode/starcoder/best_model.pt -m bigcode/starcoder --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/starcoder-woq-int4-int8-acc.log

mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/meta-llama/Meta-Llama-3.1-8B-Instruct
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/meta-llama/Meta-Llama-3.1-8B-Instruct --quant-with-amp --lowp-mode INT8 -m meta-llama/Meta-Llama-3.1-8B-Instruct --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/meta-llama/Meta-Llama-3.1-8B-Instruct/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/meta-llama/Meta-Llama-3.1-8B-Instruct/best_model.pt -m meta-llama/Meta-Llama-3.1-8B-Instruct --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/llama31-woq-int4-int8-acc.log

mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/mistralai/Mistral-7B-v0.1
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/mistralai/Mistral-7B-v0.1 --quant-with-amp --lowp-mode INT8 -m mistralai/Mistral-7B-v0.1 --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/mistralai/Mistral-7B-v0.1/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/mistralai/Mistral-7B-v0.1/best_model.pt -m mistralai/Mistral-7B-v0.1 --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/mistral-7b-woq-int4-int8-acc.log

mkdir -p /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/tiiuae/falcon-11b
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/tiiuae/falcon-11b --quant-with-amp --lowp-mode INT8 -m tiiuae/falcon-11b --low-precision-checkpoint /mnt/aitrgdata/mint/ipex24rc0/gptq/tiiuae/falcon-11b/gptq_checkpoint_g128.pt 
python single_instance/run_accuracy.py --quantized-model-path /mnt/aitrgdata/mint/ipex24rc0/woq-int4-int8/tiiuae/falcon-11b/best_model.pt -m tiiuae/falcon-11b --dtype int8 --ipex --quant-with-amp --tasks lambada_openai 2>&1 | tee -a $log_dir/falcon-11b-woq-int4-int8-acc.log
