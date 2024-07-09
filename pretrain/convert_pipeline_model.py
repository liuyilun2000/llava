
TRANSFORMERS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/cache/huggingface/transformers"
DATASETS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/cache/huggingface/datasets"
WORK_DIR = '/home/atuin/b207dd/b207dd11/'

DATASET_SIZE = 558128
NUM_DATASET_SHARDS = 437
MAP_BATCH_SIZE = 1280

llava_name = "llava-hf/llava-1.5-7b-hf"
bakllava_name = "llava-hf/bakLlava-v1-hf"
mistral_name = "mistralai/Mistral-7B-v0.1"
mixtral_name = "mistralai/Mixtral-8x7B-v0.1"
mixtral_22b_name = "mistral-community/Mixtral-8x22B-v0.1"

from PIL import Image
import requests
import json

import torch
from pathlib import Path
import os
from os.path import join
import copy
import argparse
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel



original_model_dir = '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-test01/'

model = LlavaForConditionalGeneration.from_pretrained(
    original_model_dir,
    device_map="cpu",
    torch_dtype="auto"
)

pipeline_model_dir = '/home/atuin/b207dd/b207dd11/checkpoint/global_step2465/'
#pipeline_model_dir = '/home/atuin/b207dd/b207dd11/checkpoint/global_step2275/'
pretrained_model_dir = '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-pretrained-2/'

model_static_dict = {}
for path in Path(pipeline_model_dir).iterdir():
    if not path.name.startswith('layer'):
        continue
    print(f"Processing {path}")
    layer_i = int(path.name.split('-')[0].replace('layer_', ''))
    if layer_i == 0:
        tmp_static_dict = torch.load(path, map_location="cpu")
        for k, v in tmp_static_dict.items():
            if 'multi_modal_projector' in k:
                model_static_dict[k] = v
        break
    continue

for k, v in model_static_dict.items():
    v = torch.nn.Parameter(v.to(model.device))
    a, b, c = k.rsplit('.')
    setattr(getattr(getattr(model, a), b), c, v)


# here the model is already converted and ready to be used.

pretrained_model_dir = '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-pretrained/'

model_processor = AutoProcessor.from_pretrained(pretrained_model_dir)


model.eval()
prompt = '<image>What is this?'
prompt = '<image>Render a clear and concise summary of the photo.'
prompt = '<image>Render a detailed summary of the photo.'
prompt = '<image>Describe the image concisely.'
prompt = '<image>What is the content of the image?'

prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
prompt = "<image>\nUSER: What are people doing in this image?\nASSISTANT:"
prompt = 'How is water taste like?'

prompt = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"

image_path = "/home/hpc/b207dd/b207dd11/llava/img/munich.jpg"
image = Image.open(image_path)
inputs = model_processor(text=prompt, images=image, return_tensors="pt")

inputs = model_processor(text=prompt, return_tensors="pt")
for k,v in inputs.items():
    v = v.to(model.device)

generate_ids = model.generate(**inputs, max_new_tokens=20)
model_processor.batch_decode(generate_ids)#, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]





# however directly saving the model now would make the LM part into float32 instead of bf16
# for now we just repeat the model concatenation once more as LM is untouched


mixtral_model = AutoModelForCausalLM.from_pretrained(
    mixtral_name, 
    #mixtral_22b_name, 
    torch_dtype=torch.bfloat16,
    #torch_dtype="auto",
    cache_dir=TRANSFORMERS_CACHE_DIR,
    device_map="auto"
)

new_language_model = copy.copy(mixtral_model)

new_language_model.resize_token_embeddings(32002)

model.language_model = new_language_model#.to(new_model.device)#mixtral_model



model.save_pretrained(pretrained_model_dir)




# loading


pretrained_model_dir = '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-pretrained-2/'


model = LlavaForConditionalGeneration.from_pretrained(
    pretrained_model_dir,
    device_map="auto",
    torch_dtype="auto"
)








# Lora 



from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model

from peft.tuners.tuners_utils import BaseTuner

from peft.config import PeftConfig
from peft.utils import PeftType


config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["multi_modal_projector"],
)
lora_model = get_peft_model(model, config)



    #target_modules=["block_sparse_moe"],
    #target_modules=["self_attn"],


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            #print(name)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

print_trainable_parameters(lora_model)


lora_model_dir = '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-lora/'
lora_model.save_pretrained(lora_model_dir)



config = PeftConfig.from_pretrained(lora_model_dir)
model = LlavaForConditionalGeneration.from_pretrained(config.base_model_name_or_path)
lora_model = PeftModel.from_pretrained(model, lora_model_dir, is_trainable=True)




# PEFT Tuner



config = PeftConfig(
    peft_type=PeftType.LORA, 
)

config.target_modules=["block_sparse_moe"]

BaseTuner(model, config, 'lora-ffn')







# deepseek moe

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "deepseek-ai/deepseek-moe-16b-base"
deepspeed_tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    cache_dir=TRANSFORMERS_CACHE_DIR
)
deepspeed_model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16, 
    device_map="auto", 
    trust_remote_code=True,
    cache_dir=TRANSFORMERS_CACHE_DIR
)
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
