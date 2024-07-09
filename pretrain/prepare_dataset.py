
from PIL import Image
import requests
import json
import torch

import transformers
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import Trainer, TrainingArguments
from transformers import AdamW

from torch.utils.data import DataLoader


from datasets import load_dataset
from huggingface_hub import upload_file, upload_folder
import copy
import numpy as np
import transformers
import accelerate
import pickle
from tqdm import tqdm

import peft
from peft import LoraConfig, get_peft_model



print(f"Transformers version: {transformers.__version__}")
print(f"Accelerate version: {accelerate.__version__}")
print(f"PEFT version: {peft.__version__}")


TRANSFORMERS_CACHE_DIR = "/home/vault/b207dd/b207dd11/cache/huggingface/transformers"
DATASETS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/cache/huggingface/datasets"
WORK_DIR = '/home/atuin/b207dd/b207dd11/'

#DATASET_SIZE = 558128
#NUM_DATASET_SHARDS = 437
MAP_BATCH_SIZE = 1280

llava_name = "llava-hf/llava-1.5-7b-hf"
bakllava_name = "llava-hf/bakLlava-v1-hf"
mistral_name = "mistralai/Mistral-7B-v0.1"
mixtral_name = "mistralai/Mixtral-8x7B-v0.1"


pretrained_model_dir = '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-pretrained-2/'

model_processor = AutoProcessor.from_pretrained(pretrained_model_dir)

data_dir = WORK_DIR+"LLaVA-Pretrain"
dataset = load_dataset("imagefolder", data_dir=data_dir, cache_dir=DATASETS_CACHE_DIR)
DATASET_SIZE = len(dataset['train'])
#print(dataset)

sep_token_id = model_processor.tokenizer.convert_tokens_to_ids('<s>')
eos_token_id = model_processor.tokenizer.convert_tokens_to_ids('</s>')

def preprocess_function(examples):
    images = examples['image']
    # the processor automatically adds '<s>' at the start of the sequence
    # we insert a second '<s>' manually between a and b to help locate the boundary for label
    inputs = model_processor(text=[a+'<s>'+b+'</s>' for a,b in zip(examples['text_input'],examples['text_output'])], images=images, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    new_input_ids = []
    new_attention_mask = []
    labels = []
    for i, row in enumerate(input_ids):
        # Find all indices of '<s>' tokens
        sep_indices = (row == sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_indices) > 1:
            # Start index is right after the second '<s>'
            start_index = sep_indices[-1]
            # Remove the second '<s>' token from input_ids and attention_mask
            new_row_input_ids = torch.cat((input_ids[i, :start_index], input_ids[i, start_index+1:]))
            new_input_ids.append(new_row_input_ids)
            new_attention_mask.append(torch.cat((attention_mask[i, :start_index], attention_mask[i, start_index+1:])))
            # Initialize label tensor with -100 (mask value)
            new_row_labels = torch.full_like(new_row_input_ids, -100)
            # Adjust the labels for the indices corresponding to b+'</s>'
            new_row_labels[start_index:] = new_row_input_ids[start_index:]
            labels.append(new_row_labels)
        else:
            print(f"Label Boundary Error: unexpected input structure in {examples}")
            new_input_ids.append(input_ids[i])
            new_attention_mask.append(attention_mask[i])
            labels.append(input_ids[i])
            continue
    new_input_ids = torch.stack(new_input_ids)
    new_attention_mask = torch.stack(new_attention_mask)
    labels = torch.stack(labels)
    return {
        "input_ids": new_input_ids,
        "attention_mask": new_attention_mask,
        "pixel_values": inputs["pixel_values"],
        "labels": labels
    }




skip_shard_id = 0
shard_id = 0
for shard_start in tqdm(range(0, DATASET_SIZE, MAP_BATCH_SIZE)):
    if shard_id < skip_shard_id:
        shard_id += 1
        continue
    #dataset_train = dataset['train'].shard(num_shards=NUM_DATASET_SHARDS, index=shard_id)
    shard_end = min(DATASET_SIZE, shard_start+MAP_BATCH_SIZE)
    print("Selecting range:", shard_start, '-', shard_end, "for shard", shard_id)
    dataset_train = dataset['train'].select([i for i in range(shard_start, shard_end)])
    processed_dataset = dataset_train.map(
        preprocess_function,
        batched=True, batch_size=len(dataset_train), 
        remove_columns=['image', 'text_input', 'text_output']#, num_proc=8
    )
    processed_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'pixel_values', 'labels'])    
    processed_dataset.save_to_disk(WORK_DIR+'LLaVA-Pretrain_processed_dataset/shard_'+str(shard_id))
    shard_id += 1
    break