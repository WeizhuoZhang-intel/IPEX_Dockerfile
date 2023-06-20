
import torch
import time
import json
import pathlib
import argparse

from transformers import (
    # pipeline,
    AutoModelForCausalLM,
    # AutoModel,
    #LlamaForCausalLM,
    T5ForConditionalGeneration,
    AutoTokenizer,
    #LlamaTokenizer,
)


# supported models now
MODEL_CLASSES = {
    "gpt": (AutoModelForCausalLM, AutoTokenizer),
    "opt": (AutoModelForCausalLM, AutoTokenizer),
    "bloom": (AutoModelForCausalLM, AutoTokenizer),
   # "llama": (LlamaForCausalLM, LlamaTokenizer),
    "t5": (T5ForConditionalGeneration, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
    # "chatglm": (AutoModel, AutoTokenizer),
}

# args
parser = argparse.ArgumentParser("Generation script", add_help=False)
parser.add_argument(
    "-m",
    "--model-id",
    type=str,
    default="EleutherAI/gpt-j-6B",
    help="the huggingface mdoel id"
)
parser.add_argument(
    "--device",
    type=str,
    choices=["cpu", "xpu", "cuda", "hpu"],
    default="cpu",
    help="cpu, xpu, hpu or cuda",
)
parser.add_argument(
    "--dtype",
    type=str,
    choices=["float32", "bfloat16", "float16"],
    default="bfloat16",
    help="bfloat16, float32 or float16",
)
parser.add_argument(
    "--input-tokens", default="32", type=str, help="input tokens length if needed from prompt.json"
)
parser.add_argument(
    "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument(
    "--prompt", default=None, type=str, help="input prompt for self-defined if needed"
)
parser.add_argument("--greedy", action="store_true")
parser.add_argument("--ipex", action="store_true")
parser.add_argument("--ipex_tpp", action="store_true", help="enable tpp optimization for ipex bfloat16 only")
parser.add_argument("--jit", action="store_true")
parser.add_argument("--num-iter", default=100, type=int, help="num iter")
parser.add_argument("--num-warmup", default=10, type=int, help="num warmup")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument("--token-latency", action="store_true", help="get token latency")
args = parser.parse_args()
print(args)


# device
device = torch.device(args.device)

# import extension
if args.ipex:
    import intel_extension_for_pytorch as ipex
    try:
        ipex._C.disable_jit_linear_repack()
    except Exception:
        pass
if args.device == "hpu":
    import habana_frameworks.torch as ht

# dtype
amp_enabled = True if args.dtype != "float32" else False
amp_dtype = getattr(torch, args.dtype)

# load model
model_type = next((x for x in MODEL_CLASSES.keys() if x in args.model_id.lower()), 'auto')
model_class = MODEL_CLASSES[model_type]
model = model_class[0].from_pretrained(
    args.model_id, low_cpu_mem_usage=True, return_dict=not args.jit, torch_dtype=amp_dtype
)
tokenizer = model_class[1].from_pretrained(args.model_id)
model = model.eval().to(device)
model = model.to(memory_format=torch.channels_last)

# to hpu graph
if args.device == "hpu":
    model = ht.hpu.wrap_in_hpu_graph(model)
# to ipex
if args.ipex:
    # tpp only use for gptj bfloat16
    ipex_tpp_enabled = args.ipex_tpp and args.dtype == "bfloat16" and model.config.model_type == "gptj"
    model = ipex.optimize(model.eval(), dtype=amp_dtype, inplace=True, weights_prepack=not ipex_tpp_enabled)
    if ipex_tpp_enabled:
        ipex.tpp.Apply_TPP_optimization(model, dtype=torch.bfloat16)
        print("---- Use IPEX TPP optimizaiton")

# input prompt
current_path = pathlib.Path(__file__).parent.resolve()
with open(str(current_path) + "/prompt.json") as f:
    prompt_pool = json.load(f)
if args.prompt is not None:
    prompt = args.prompt
elif model_type == "auto":
    raise SystemExit("[ERROR] model prompt is not supported, please use --prompt for this model: " + args.model_id)
elif args.input_tokens in prompt_pool[model_type]:
    prompt = prompt_pool[model_type][args.input_tokens]
else:
    raise SystemExit("[ERROR] Plese use --prompt if want to use custom input.")

input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
print("---- Prompt size:", input_size)

# generate args
generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=1 if args.greedy else 4)
if args.jit:
    torch._C._jit_set_texpr_fuser_enabled(False)
    generate_kwargs["jit"] = True
if args.token_latency:
    generate_kwargs["token_latency"] = True

# start
total_time = 0.0
num_iter = args.num_iter
num_warmup = args.num_warmup
prompt = [prompt] * args.batch_size
total_list = []
with torch.inference_mode(), torch.no_grad(), torch.autocast(
    device_type=args.device,
    enabled=amp_enabled,
    dtype=amp_dtype if amp_enabled else None,
):
    for i in range(num_iter):
        tic = time.time()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        output = model.generate(
            input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs
        )
        gen_ids = output[0] if args.token_latency else output
        gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        if args.device == "xpu":
            torch.xpu.synchronize()
        elif args.device == "cuda":
            torch.cuda.synchronize()
        elif args.device == "hpu":
            gen_ids.to("cpu")
        toc = time.time()
        input_tokens_lengths = [x.shape[0] for x in input_ids]
        output_tokens_lengths = [x.shape[0] for x in gen_ids]
        total_new_tokens = [o - i if model.config.model_type != 't5' else o for i, o in zip(input_tokens_lengths, output_tokens_lengths)]
        print(gen_text, total_new_tokens, flush=True)
        if model.config.model_type != 't5':
            assert total_new_tokens[0] == args.max_new_tokens, "Generated new tokens != max new tokens"
        if i >= num_warmup:
            total_time += toc - tic
            if args.token_latency:
                total_list.append(output[1])

print("\n", "-" * 10, "Summary:", "-" * 10)
latency = total_time / (num_iter - num_warmup)
print("Inference latency: %.3f sec." % latency)
if args.token_latency:
    import numpy as np
    from itertools import chain
    first_latency = np.mean([x[0] for x in total_list])
    average_2n = list(chain(*[x[1:] for x in total_list]))
    average_2n.sort()
    average_2n_latency = np.mean(average_2n)
    p90_latency = average_2n[int(len(average_2n) * 0.9)]
    p99_latency = average_2n[int(len(average_2n) * 0.99)]
    print("First token average latency: %.3f sec." % first_latency)
    print("Average 2... latency: %.3f sec." % average_2n_latency)
    print("P90 2... latency: %.3f sec." % p90_latency)
    print("P99 2... latency: %.3f sec." % p99_latency)
