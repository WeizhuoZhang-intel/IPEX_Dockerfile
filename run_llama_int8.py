
import os
import psutil
import argparse
import time
import json
from pathlib import Path
import pathlib

from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer

import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader

import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import convert, prepare


parser = argparse.ArgumentParser('LLaMA generation script', add_help=False)
parser.add_argument(
    '-m', '--model-id', default=None, type=str, required=True, help="your llama model"
)
parser.add_argument(
    "--device",
    type=str,
    choices=["cpu", "cuda"],
    help="cpu or cuda",
    default="cpu",
)
parser.add_argument("--dtype", type=str, default="int8")
parser.add_argument(
    "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument("--dataset", nargs="?", default="lambada", const="lambada")
parser.add_argument("--split", nargs="?", default="validation", const="validation")
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument("--ipex_dynamic_quantize", action="store_true")
parser.add_argument("--ipex_static_quantize", action="store_true")
parser.add_argument("--quantize_with_inc", action="store_true")
parser.add_argument("--ipex_smooth_quant", action="store_true")
parser.add_argument("--ipex", action="store_true", help="no use and enabled always")
parser.add_argument("--jit", action="store_true")
parser.add_argument("--int8", action="store_true")
parser.add_argument(
    "--int8_bf16_mixed",
    action="store_true",
    help="by default it is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
)
parser.add_argument("--quantized_model_path", default="./saved_result/best_model.pt")
parser.add_argument("--lambada", action="store_true")
parser.add_argument("--accuracy_only", action="store_true")
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--input-tokens", default="32", type=str)
parser.add_argument("--prompt", default=None, type=str)
parser.add_argument("--num-iter", default=100, type=int, help="num iter")
parser.add_argument("--num-warmup", default=10, type=int, help="num warmup")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument("--token-latency", action="store_true")
parser.add_argument("--greedy", action="store_true")
args = parser.parse_args()


# disable
try: ipex._C.disable_jit_linear_repack()
except Exception: pass

device = torch.device(args.device)
args.dtype = "int8" if args.int8 or args.int8_bf16_mixed else args.dtype

# amp autocast
if args.int8_bf16_mixed:
    amp_enabled = True
    amp_dtype = torch.bfloat16
else:
    amp_enabled = False
    amp_dtype = torch.float32


num_beams=1 if args.greedy else 4
generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=num_beams)


# load model
user_model = LlamaForCausalLM.from_pretrained(args.model_id, low_cpu_mem_usage=True, torchscript=args.jit)
tokenizer = LlamaTokenizer.from_pretrained(args.model_id)
print("Data type of the model:", user_model.dtype)


beam_idx_tmp = torch.zeros((2048, int(args.batch_size * num_beams)), dtype=torch.long).contiguous()
global_past_key_value = [(torch.zeros([1,user_model.config.num_attention_heads,1,int(user_model.config.hidden_size/user_model.config.num_attention_heads)]).contiguous(),
                           torch.zeros([1,user_model.config.num_attention_heads,1,int(user_model.config.hidden_size/user_model.config.num_attention_heads)]).contiguous(), beam_idx_tmp, torch.zeros(1, dtype=torch.long).contiguous()) for i in range(user_model.config.num_hidden_layers)]


class Evaluator:
    def __init__(self, dataset, tokenizer, batch_size=8, pad_val=1, pad_max=196):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.pad_val = pad_val
        self.pad_max = pad_max

        # tokenize the dataset
        self.dataset = self.dataset.map(self.tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids"])

    @torch.no_grad()
    def tokenize_function(self, examples):
        example = self.tokenizer(examples["text"])
        return example

    @torch.no_grad()
    def collate_batch(self, batch):
        position_ids_padded = []
        input_ids_padded = []
        last_ind = []
        attention_mask_padded = []
        for text in batch:
            # we cut the sentence if it exceeds pad_max, we are using tuned max 196 from gptj model; TODO: tune best pad_max 
            input_ids = text["input_ids"] if text["input_ids"].shape[0] <= self.pad_max else text["input_ids"][0:int(self.pad_max-1)]
            pad_len = self.pad_max - input_ids.shape[0]
            last_ind.append(input_ids.shape[0] - 1)
            attention_mask = torch.ones(len(input_ids))
            position_ids = torch.arange(len(input_ids))
            input_ids = pad(input_ids, (0, pad_len), value=self.pad_val)
            input_ids_padded.append(input_ids)
            attention_mask = pad(attention_mask, (0, pad_len), value=0)
            attention_mask_padded.append(attention_mask)
            position_ids = pad(position_ids, (0, pad_len), value=self.pad_val)
            position_ids_padded.append(position_ids)
        return (
            (
                torch.vstack(input_ids_padded),
                torch.vstack(attention_mask_padded),
                torch.vstack(position_ids_padded),
                tuple(global_past_key_value),
            ),
            torch.tensor(last_ind),
        )

    @torch.no_grad()
    def evaluate(self, model):

        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        latency = 0
        test_dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
        )

        for i, (
            (input_ids, attention_mask, position_ids, past_key_values),
            last_ind,
        ) in enumerate(test_dataloader):
            label = input_ids[torch.arange(len(last_ind)), last_ind]
            input_ids[torch.arange(len(last_ind)), last_ind] = self.pad_val
            pad_len = self.pad_max - last_ind - 1
            start = time.time()

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                position_ids= position_ids,
                past_key_values=past_key_values,
            )

            latency += time.time() - start

            last_token_logits = outputs[0][torch.arange(len(last_ind)), -2 - pad_len, :]

            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            if i % 50 == 0:
                print(hit / total)
                print("Processed minibatch:", i)

        acc = hit / total
        print(acc)
        lantecy = latency / len(self.dataset)
        return acc, lantecy

if args.lambada:
    full_dataset = load_dataset(args.dataset)
    dataset = full_dataset["validation"]
    calib_dataset = full_dataset["train"]

    user_model.eval()
    evaluator = Evaluator(dataset, tokenizer, args.batch_size)
    calib_evaluator = Evaluator(calib_dataset, tokenizer, args.batch_size)

    calib_dataloader = DataLoader(
        calib_evaluator.dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=evaluator.collate_batch,
    )

    test_dataloader = DataLoader(
        evaluator.dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=evaluator.collate_batch,
    )

def calib_func(prepared_model):
    for i, (
        (input_ids, position_ids, attention_mask, past_key_values),
        last_ind,
    ) in enumerate(calib_dataloader):
        if i == 8:
            break
        prepared_model(
            input_ids,
            position_ids = position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )


def eval_func(traced_model):
    acc, latency = evaluator.evaluate(traced_model)
    print("Accuracy:", acc)
    print("Latency (sec):", latency)
    return acc


if args.jit:
    torch._C._jit_set_texpr_fuser_enabled(False)
    generate_kwargs["jit"] = True
    if args.int8 or args.int8_bf16_mixed:
        generate_kwargs["ipex_int8"] = True
        generate_kwargs["quantized_model_path"] = args.quantized_model_path


if args.ipex_dynamic_quantize:
    example_inputs=None
    input_ids = torch.ones(32).to(torch.long)
    attention_mask = torch.ones(len(input_ids) + 1)
    attention_mask[0] = 0
    last_ind = input_ids.shape[0] - 1
    position_ids = torch.arange(len(input_ids))
    example_inputs=(input_ids.unsqueeze(0), attention_mask.unsqueeze(0), position_ids.unsqueeze(0), tuple(global_past_key_value))

    from intel_extension_for_pytorch.quantization import prepare, convert
    qconfig = ipex.quantization.default_dynamic_qconfig
    prepared_model = prepare(user_model.eval(), qconfig, example_inputs=example_inputs)
    with torch.no_grad():
        convert_model = convert(prepared_model.eval()).eval()
        self_jit = torch.jit.trace(convert_model.eval(), example_inputs, strict=False)
        self_jit = torch.jit.freeze(self_jit.eval())
        self_jit.save(args.output_dir+"/ipex_dynamic_quantize_model.pt")

if args.ipex_static_quantize or args.ipex_smooth_quant:
    example_inputs=None
    for i, (
        (input_ids, attention_mask, position_ids, past_key_values),
        last_ind,
    ) in enumerate(calib_dataloader):
        example_inputs=(input_ids, attention_mask, position_ids, past_key_values)
        break
    from intel_extension_for_pytorch.quantization import prepare, convert
    qconfig = ipex.quantization.default_static_qconfig
    if args.ipex_smooth_quant:
        # qconfig = ipex.quantization.get_smooth_quant_static_qconfig()
        qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping()
    prepared_model = prepare(user_model.eval(), qconfig, example_inputs=example_inputs)
    with torch.no_grad():
        for i, (
            (input_ids, attention_mask, position_ids, past_key_values),
            last_ind,
        ) in enumerate(calib_dataloader):
            if i == 8:
                break
            prepared_model(
                input_ids,
                position_ids = position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values)
    with torch.no_grad(), torch.autocast(device_type=args.device, enabled=amp_enabled or args.int8_bf16_mixed, dtype=amp_dtype):
        convert_model = convert(prepared_model.eval()).eval()
        # if args.ipex_smooth_quant:
        #     convert_model(*example_inputs)
        self_jit = torch.jit.trace(convert_model.eval(), example_inputs, strict=False)
        self_jit = torch.jit.freeze(self_jit.eval())
        self_jit.save(args.output_dir+"/ipex_static_quantize_model.pt")

if args.quantize_with_inc:
    from neural_compressor import PostTrainingQuantConfig, quantization

    op_type_dict = {
        "add": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},
        "linear": {
            "weight": {
                "dtype": ["int8"],
                "scheme": ["sym"],
                "granularity": ["per_channel"],
                "algorithm": ["minmax"],
            },
            "activation": {
                "dtype": ["uint8"],
                "scheme": ["asym"],
                "granularity": ["per_tensor"],
                "algorithm": ["kl"],
            },
        },
    }

    excluded_precisions = [] if args.int8_bf16_mixed else ["bf16"]
    conf = PostTrainingQuantConfig(
        backend="ipex",
        excluded_precisions=excluded_precisions,
        op_type_dict=op_type_dict,
    )
    q_model = quantization.fit(
        user_model,
        conf,
        calib_dataloader=calib_dataloader,
        eval_func=eval_func,
        calib_func=calib_func,
    )


    q_model.save(args.output_dir)

if args.accuracy_only:
    # backend setting for dynamic quantization: torch.backends.quantized.engine = 'onednn', default is fbgemm
    if args.int8 or args.int8_bf16_mixed:
        user_model = torch.jit.load(
            args.quantized_model_path
        )
        user_model = torch.jit.freeze(user_model.eval())

    if args.jit and (args.dtype == "bfloat16" or args.dtype == "float32"):
        input_ids = torch.ones(32).to(torch.long)
        attention_mask = torch.ones(len(input_ids) + 1)
        attention_mask[0] = 0
        last_ind = input_ids.shape[0] - 1
        position_ids = torch.arange(len(input_ids))
        example_inputs=(input_ids.unsqueeze(0), attention_mask.unsqueeze(0), position_ids.unsqueeze(0), tuple(global_past_key_value))
        with torch.no_grad(), torch.autocast(device_type=args.device, enabled=amp_enabled, dtype=amp_dtype):
            user_model = torch.jit.trace(user_model.eval(), example_inputs, strict=False)
            user_model = torch.jit.freeze(user_model.eval())

    with torch.autocast(device_type=args.device, enabled=amp_enabled or args.int8_bf16_mixed, dtype=amp_dtype):
      eval_func(user_model)

if args.benchmark:
    # input prompt
    current_path = pathlib.Path(__file__).parent.resolve()
    with open(str(current_path) + '/prompt.json') as f:
        prompt_pool = json.load(f)
    if args.prompt is not None:
        prompt = args.prompt
    elif args.input_tokens in prompt_pool["llama"]:
        prompt = prompt_pool["llama"][args.input_tokens]
    else:
        raise SystemExit('[ERROR] Plese use --prompt if want to use custom input.')

    input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
    print("---- Prompt size:", input_size)
    if args.token_latency:
        generate_kwargs["token_latency"] = True

    total_time = 0.0
    num_iter = args.num_iter
    num_warmup = args.num_warmup
    prompt = [prompt] * args.batch_size
    total_list = []
    with torch.inference_mode(), torch.no_grad(), torch.autocast(
        device_type=args.device,
        enabled=amp_enabled or args.int8_bf16_mixed,
        dtype= torch.bfloat16 if args.int8_bf16_mixed else None
    ):
       for i in range(num_iter):
            tic = time.time()
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            output = user_model.generate(
                input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs
            )
            gen_ids = output[0] if args.token_latency else output
            gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            if args.device == "cuda":
                torch.cuda.synchronize()
            toc = time.time()
            input_tokens_lengths = [x.shape[0] for x in input_ids]
            output_tokens_lengths = [x.shape[0] for x in gen_ids]
            total_new_tokens = [o - i if user_model.config.model_type != 't5' else o for i, o in zip(input_tokens_lengths, output_tokens_lengths)]
            print(gen_text, total_new_tokens, flush=True)
            # if user_model.config.model_type != 't5':
            #     assert total_new_tokens[0] == args.max_new_tokens, "Generated new tokens != max new tokens"
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
