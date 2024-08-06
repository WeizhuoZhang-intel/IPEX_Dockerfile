#!/bin/bash

set -x
pwd


# python utils/create_shard_model.py -m meta-llama/Llama-2-13b-hf --save-path /unipath/shard/meta-llama/Llama-2-13b-hf

# sleep 3s

python utils/create_shard_model.py -m meta-llama/Llama-2-70b-hf --save-path /unipath/shard/meta-llama/Llama-2-70b-hf

sleep 3s

python utils/create_shard_model.py -m tiiuae/falcon-40b --save-path /unipath/shard/tiiuae/falcon-40b

sleep 3s

python utils/create_shard_model.py -m facebook/opt-1.3b --save-path /unipath/shard/facebook/opt-1.3b

sleep 3s 

python utils/create_shard_model.py -m facebook/opt-30b --save-path /unipath/shard/facebook/opt-30b

sleep 3s 

python utils/create_shard_model.py -m baichuan-inc/Baichuan2-13B-Chat --save-path /unipath/shard/baichuan-inc/Baichuan2-13B-Chat
sleep 3s 
python utils/create_shard_model.py -m baichuan-inc/Baichuan2-7B-Chat --save-path /unipath/shard/baichuan-inc/Baichuan2-7B-Chat
sleep 3s 
python utils/create_shard_model.py -m baichuan-inc/Baichuan-13B-Chat --save-path /unipath/shard/baichuan-inc/Baichuan-13B-Chat

sleep 3s 

python utils/create_shard_model.py -m databricks/dolly-v2-12b --save-path /unipath/shard/databricks/dolly-v2-12b

sleep 3s 

python utils/create_shard_model.py -m EleutherAI/gpt-neox-20b --save-path /unipath/shard/EleutherAI/gpt-neox-20b

sleep 3s 

python utils/create_shard_model.py -m Salesforce/codegen-2B-multi --save-path /unipath/shard/Salesforce/codegen-2B-multi

sleep 3s 

python utils/create_shard_model.py -m bigcode/starcoder --save-path /unipath/shard/bigcode/starcoder

sleep 3s

python utils/create_shard_model.py -m Salesforce/codegen-2B-multi --save-path /unipath/shard/Salesforce/codegen-2B-multi

sleep 3s






python utils/create_shard_model.py -m google/flan-t5-xl --save-path /unipath/shard/google/flan-t5-xl

sleep 3s

python utils/create_shard_model.py -m mistralai/Mistral-7B-v0.1 --save-path /unipath/shard/mistralai/Mistral-7B-v0.1

sleep 3s
python utils/create_shard_model.py -m mosaicml/mpt-7b --save-path /unipath/shard/mosaicml/mpt-7b
sleep 3s
python utils/create_shard_model.py -m mistralai/Mixtral-8x7B-v0.1 --save-path /unipath/shard/mistralai/Mixtral-8x7B-v0.1
sleep 3s
python utils/create_shard_model.py -m Qwen/Qwen-7B-Chat --save-path /unipath/shard/Qwen/Qwen-7B-Chat
sleep 3s

python utils/create_shard_model.py -m microsoft/phi-2 --save-path /unipath/shard/microsoft/phi-2
sleep 3s
python utils/create_shard_model.py -m microsoft/Phi-3-mini-128k-instruct --save-path /unipath/shard/microsoft/Phi-3-mini-128k-instruct
sleep 3s
python utils/create_shard_model.py -m microsoft/Phi-3-mini-4k-instruct --save-path /unipath/shard/microsoft/Phi-3-mini-4k-instruct
sleep 3s
python utils/create_shard_model.py -m microsoft/Phi-3-medium-4k-instruct --save-path /unipath/shard/microsoft/Phi-3-medium-4k-instruct
sleep 3s
python utils/create_shard_model.py -m microsoft/Phi-3-medium-128k-instruct --save-path /unipath/shard/microsoft/Phi-3-medium-128k-instruct
sleep 3s
python utils/create_shard_model.py -m meta-llama/Meta-Llama-3-8B --save-path /unipath/shard/meta-llama/Meta-Llama-3-8B
sleep 3s
python utils/create_shard_model.py -m meta-llama/Meta-Llama-3-70B --save-path /unipath/shard/meta-llama/Meta-Llama-3-70B



