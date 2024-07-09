
TRANSFORMERS_CACHE_DIR = "/home/vault/b207dd/b207dd11/cache/huggingface/transformers"
DATASETS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/cache/huggingface/datasets"
WORK_DIR = '/home/atuin/b207dd/b207dd11/'
pretrained_model_dir = '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-pretrained/'


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
from transformers import LlavaConfig, LlavaForConditionalGeneration
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel



pretrained_model_dir = '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-pretrained/'

model = LlavaForConditionalGeneration.from_pretrained(
    pretrained_model_dir,
    device_map="auto",
    torch_dtype="auto"
)




from llava.mixtral_modification.configuration_mixtral import MixtralAdapterConfig
from llava.mixtral_modification.modeling_mixtral import MixtralAdapterModel, MixtralAdapterForCausalLM



pretrained_model_dir = '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-pretrained/'

llava_config = LlavaConfig.from_pretrained(pretrained_model_dir)
llava_config.text_config = mixtral_config


model = LlavaForConditionalGeneration.from_pretrained(
    pretrained_model_dir,
    config=llava_config,
    device_map="auto",
    torch_dtype="auto"
)