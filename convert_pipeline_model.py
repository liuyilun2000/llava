
TRANSFORMERS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/cache/huggingface/transformers"
DATASETS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/cache/huggingface/datasets"
WORK_DIR = '/home/atuin/b207dd/b207dd11/'

pretrained_model_dir = '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-pretrained-2/'


llava_name = "llava-hf/llava-1.5-7b-hf"
bakllava_name = "llava-hf/bakLlava-v1-hf"
mistral_name = "mistralai/Mistral-7B-v0.1"
mixtral_name = "mistralai/Mixtral-8x7B-v0.1"
mixtral_22b_name = "mistral-community/Mixtral-8x22B-v0.1"


device = 'auto'

import random
random.seed(42)

import os
import copy
import json
import math
import time
import pickle
import argparse
import requests
import numpy as np
from tqdm import tqdm
from pathlib import Path

from PIL import Image

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from datasets import load_dataset, load_from_disk
#from huggingface_hub import upload_file, upload_folder

import transformers
from transformers import LlavaConfig, LlavaForConditionalGeneration
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel



from llava.mixtral_modification.configuration_mixtral import MixtralAdapterConfig
from llava.mixtral_modification.modeling_mixtral import MixtralAdapterModel, MixtralAdapterForCausalLM

AutoConfig.register("mixtral-adapter", MixtralAdapterConfig)
AutoModel.register(MixtralAdapterConfig, MixtralAdapterModel)
AutoModelForCausalLM.register(MixtralAdapterConfig, MixtralAdapterForCausalLM)



import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
from deepspeed.runtime.pipe import ProcessTopology
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed import comm as dist

from deepspeed_pipeline_model import *


from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model

from peft.tuners.tuners_utils import BaseTuner

from peft.config import PeftConfig
from peft.utils import PeftType


from safetensors.torch import save_file, safe_open



def load_config_from_file(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config



def set_args(config):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--num_stages", type=int, default=8, help="")
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    #
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--multi_modal_projector_pretraining", type=bool, default=True, help="")
    #parser.add_argument("--model_name_or_path", type=str, help="", required=True)
    parser.add_argument("--num_train_epochs", type=int, default=1, help="")
    #
    parser.add_argument("--shared_adapter_type", type=str, default="", help="")
    parser.add_argument("--shared_adapter_num", type=int, default=0, help="")
    parser.add_argument("--lora_r", type=int, default=0, help="")
    parser.add_argument("--lora_alpha", type=int, default=0, help="")
    parser.add_argument("--hidden_dim", type=int, default=0, help="")
    #
    parser.add_argument("--dataloader_batch_size", type=int, default=1, help="")
    parser.add_argument("--train_batch_size", type=int, default=128, help="")
    parser.add_argument("--train_micro_batch_size_per_gpu", type=int, default=1, help="")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128, help="")
    #
    parser.add_argument("--steps_per_print", default=1, type=int, help="")
    parser.add_argument("--save_model_shard", default=20, type=int, help="")
    parser.add_argument("--skip_shard", default=0, type=int, help="")
    parser.add_argument("--checkpoint_dir", type=str, default=WORK_DIR+"checkpoint", help="")
    #parser = deepspeed.add_config_arguments(parser)
    parser.set_defaults(**config)
    return parser.parse_args()



#model_name = 'LLaVA-PEFT_adapter_lora_32_64'
model_name = 'LLaVA-PEFT_adapter_lora_32_64_4'
model_name = 'LLaVA-PEFT_adapter_32_64_4'
model_name = 'LLaVA-PEFT_adapter_32_64'
pretrained_model_dir = '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-pretrained-2/'
pipeline_model_dir = f'/home/atuin/b207dd/b207dd11/{model_name}_checkpoint/'

checkpoint_name = 'save_step1091'
#checkpoint_name = 'save_step2582'
#checkpoint_name = 'global_step1047'
checkpoint_dir = pipeline_model_dir + checkpoint_name
config_file_path = pipeline_model_dir + 'config.json'

config = load_config_from_file(config_file_path)

args = set_args(config)
print(args)

if args.shared_adapter_type == 'LoRA':
    shared_adapter_args = {'r': args.lora_r, 'lora_alpha': args.lora_alpha, 'lora_dropout': 0.1}
elif args.shared_adapter_type == "Parallel_Adapter":
    shared_adapter_args = {'hidden_dim': args.hidden_dim, 'hidden_act': 'silu', 'dropout': 0.1}


mixtral_config = MixtralAdapterConfig(
    vocab_size=32002,
    shared_adapter=True,
    shared_adapter_type=args.shared_adapter_type,
    shared_adapter_num=args.shared_adapter_num,
    shared_adapter_args=shared_adapter_args,
#        shared_routing_adapter=True,
#        shared_routing_adapter_num_experts=4,
#        shared_routing_adapter_num_experts_per_tok=2,
#        shared_routing_adapter_type='LoRA',
#        shared_routing_adapter_args={'r': 32, 'lora_alpha': 32, 'lora_dropout': 0.1},
    output_router_logits=False
)


llava_config = LlavaConfig.from_pretrained(pretrained_model_dir)
llava_config.text_config = mixtral_config


model = LlavaForConditionalGeneration.from_pretrained(
    pretrained_model_dir,
    config=llava_config,
    device_map=device,
    torch_dtype="auto"
)




trainable_param_names = ['lora_A', 'lora_B', 'adapter_w1', 'adapter_w2']

multi_modal_projector_static_dict = {}
adapter_static_dict = {}
for path in Path(checkpoint_dir).iterdir():
    if not path.name.startswith('layer'):
        continue
    print(f"Processing {path}")
    layer_i = int(path.name.split('-')[0].replace('layer_', ''))
    tmp_static_dict = torch.load(path, map_location="cpu")
    for k, v in tqdm(tmp_static_dict.items()):
        if 'multi_modal_projector' in k:
            multi_modal_projector_static_dict[k] = v
        if any(param_name in k for param_name in trainable_param_names) and not ('vision' in k):
            key_parts = k.split('.')
            if key_parts[0] == 'layers' and key_parts[1].isdigit():
                pipeline_layer_num = int(key_parts[1])
                real_layer_num = pipeline_layer_num + (layer_i-1) * 4
                key_parts[1] = str(real_layer_num)
                new_key = '.'.join(key_parts)
                adapter_static_dict[new_key] = v
    else: 
        continue



save_dir = f"/home/vault/b207dd/b207dd11/llava-mixtral/{model_name}/{checkpoint_name}"

save_file(multi_modal_projector_static_dict, save_dir+"/multi_modal_projector.safetensors")
save_file(adapter_static_dict, save_dir+"/lora.safetensors")






def find_key_substring_in_name(name, keys_dict):
    for key in keys_dict.keys():
        if key in name:
            return key
    return None


mm_count = 0
adapter_count = 0
for name, param in model.named_parameters():
    if 'vision_tower' in name:
        continue
    found_key_multi_modal = find_key_substring_in_name(name, multi_modal_projector_static_dict)
    found_key_lora = find_key_substring_in_name(name, adapter_static_dict)
    if found_key_multi_modal:
        print(f"Name: {name}, Found Key in multi_modal_projector_static_dict: {found_key_multi_modal}")
        model.state_dict()[name].copy_(multi_modal_projector_static_dict[found_key_multi_modal])
        mm_count += 1
    elif found_key_lora:
        print(f"Name: {name}, Found Key in adapter_static_dict: {found_key_lora}")
        model.state_dict()[name].copy_(adapter_static_dict[found_key_lora])
        adapter_count += 1

print(mm_count, adapter_count)





### lora load

save_dir = f"/home/vault/b207dd/b207dd11/llava-mixtral/{model_name}/{checkpoint_name}"

multi_modal_projector_static_dict = {}
with safe_open(save_dir+"/multi_modal_projector.safetensors", framework="pt", device='cpu') as f:
    for k in f.keys():
        multi_modal_projector_static_dict[k] = f.get_tensor(k)

adapter_static_dict = {}
with safe_open(save_dir+"/lora.safetensors", framework="pt", device='cpu') as f:
    for k in f.keys():
        adapter_static_dict[k] = f.get_tensor(k)






def find_key_substring_in_name(name, keys_dict):
    for key in keys_dict.keys():
        if key in name:
            return key
    return None


mm_count = 0
adapter_count = 0
for name, param in model.named_parameters():
    if 'vision_tower' in name:
        continue
    found_key_multi_modal = find_key_substring_in_name(name, multi_modal_projector_static_dict)
    found_key_lora = find_key_substring_in_name(name, adapter_static_dict)
    if found_key_multi_modal:
        print(f"Name: {name}, Found Key in multi_modal_projector_static_dict: {found_key_multi_modal}")
        model.state_dict()[name].copy_(multi_modal_projector_static_dict[found_key_multi_modal])
        mm_count += 1
    elif found_key_lora:
        print(f"Name: {name}, Found Key in adapter_static_dict: {found_key_lora}")
        model.state_dict()[name].copy_(adapter_static_dict[found_key_lora])
        adapter_count += 1

print(mm_count, adapter_count)




# here the model is already converted and ready to be used.

pretrained_model_dir = '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-pretrained-2/'

model_processor = AutoProcessor.from_pretrained(pretrained_model_dir)


model.eval()




def generate_conversation(msg_hist, msg_queue, image, 
            add_hist_tokens=False,
            add_generation_prompt=True, additional_prompt="",
            verbose=False, 
            max_new_tokens=32, output_hist=True):
    ans_list = []
    for msg in (tqdm(msg_queue) if verbose else msg_queue):
        msg_hist.append(
            {"role": "user", "content": msg}
        )
        text = model_processor.tokenizer.apply_chat_template(
            msg_hist, 
            add_generation_prompt=add_generation_prompt, 
            tokenize=False
        )
        text += additional_prompt
        inputs = model_processor(
            text=text, 
            images=image, 
            return_tensors="pt"
        )
        for k,v in inputs.items():
            if v is None:
                continue
            v = v.to(model.device)
        '''
        terminators = [
            model_processor.tokenizer.eos_token_id,
            model_processor.tokenizer.convert_tokens_to_ids('\n')
        ]
        '''
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            #eos_token_id=terminators
        )
        ans = model_processor.batch_decode(  
            generate_ids if add_hist_tokens else generate_ids[:,inputs.input_ids.shape[1]:],
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )[0]
        msg_hist.append(
            {"role": "assistant", "content": ans}
        )
        ans_list.append(ans)
        if verbose:
            print(msg_hist)
    if output_hist:
        return msg_hist
    else:
        return ans_list


system_message = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
messages = [
    {"role": "system", "content": system_message}
]
image_path = "/home/hpc/b207dd/b207dd11/llava/img/olympic.jpg"
image = Image.open(image_path)
user_messages = [
    "<image>\nWhat can be seen in the image?",
    "Which countries' flags are shown in the image?",
    "How many people are there?"
]
generate_conversation(msg_hist=messages, msg_queue=user_messages, image=image, verbose=True, max_new_tokens=100)





system_message = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
messages = [
    {"role": "system", "content": system_message}
]
image_path = "/home/hpc/b207dd/b207dd11/llava/img/food.jpg"
image = Image.open(image_path)
user_messages = [
    "<image>\nWhat can be found in the image?",
    "Which dish would you recommend?",
    "Where and what color are the chopsticks?",
]
generate_conversation(msg_hist=messages, msg_queue=user_messages, image=image, verbose=True, max_new_tokens=100)

 
















'''
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    cache_dir=TRANSFORMERS_CACHE_DIR,
    token='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'
)
chat_template = tokenizer.chat_template


'''

new_chat_template = """
{% if not add_generation_prompt is defined %}
    {% set add_generation_prompt = false %}
{% endif %}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content'] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set loop_messages = messages %}
{%- endif %}

{{- bos_token }}
{%- for message in loop_messages %}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}
    {%- endif %}
    {%- if message['role'] == 'user' %}
        {%- if loop.first and system_message is defined %}
            {{- system_message + '###Human: ' + message['content']}}
        {%- else %}
            {{- '###Human: ' + message['content'] }}
        {%- endif %}
    {%- elif message['role'] == 'assistant' %}
        {{- '###Assistant: ' + message['content'] + eos_token}}
    {%- else %}
        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}
    {%- endif %}
{%- endfor %}
{% if add_generation_prompt %}
    {{- '###Assistant: ' }}
{% endif %}"""

model_processor.tokenizer.chat_template = new_chat_template

'''