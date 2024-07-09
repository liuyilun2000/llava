#CUDA_VISIBLE_DEVICES=0,1,2,3 python3
#CUDA_VISIBLE_DEVICES=4,5,6,7 python3

from PIL import Image
import requests
import json
import torch

import transformers
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import Trainer, TrainingArguments
from transformers import AdamW

from torch.utils.data import DataLoader

import datasets
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

llava_name = "llava-hf/llava-1.5-7b-hf"
bakllava_name = "llava-hf/bakLlava-v1-hf"
mistral_name = "mistralai/Mistral-7B-v0.1"
mixtral_name = "mistralai/Mixtral-8x7B-v0.1"


dataset = load_dataset("lmms-lab/COCO-Caption", cache_dir=DATASETS_CACHE_DIR, split='test[:50%]')
test = dataset.select_columns(['question_id', 'image', 'question'])



dataset = load_dataset("lmms-lab/COCO-Caption", cache_dir=DATASETS_CACHE_DIR, split='test[50%:]')
test = dataset.select_columns(['question_id', 'image', 'question']) #[20388:]


'''
_IMAGES_URLS = {
    "train": "http://images.cocodataset.org/zips/train2014.zip",
    "validation": "http://images.cocodataset.org/zips/val2014.zip",
    "test": "http://images.cocodataset.org/zips/test2014.zip"
}

_SPLIT_MAP = {"test": "test2014"}#"train": "train2014", "validation": "val2014"}
'''


model_processor = AutoProcessor.from_pretrained(
    bakllava_name, 
    cache_dir=TRANSFORMERS_CACHE_DIR
)

pretrained_model_dir = '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-pretrained/'

model = LlavaForConditionalGeneration.from_pretrained(
    pretrained_model_dir,
    device_map="auto",
    torch_dtype="auto"
)



#print(dataset)

def append_result_to_file(result, file_path):
    """Appends a single result to a JSON file."""
    with open(file_path, 'a') as file:
        json.dump(result, file)
        file.write(',\n')  # Add a newline to separate JSON objects

file_path = '/home/hpc/b207dd/b207dd11/llava/captions_2.json'

#open(file_path, 'w').close()



last_image_id = 579808



# Process each item in the test dataset
for item in tqdm(test):
    image_id = int(item['question_id'].split('_')[-1].split('.')[0])
    if image_id<=last_image_id:
        continue
    prompt = item['question'] + "<image>"
    image = item['image']
    inputs = model_processor(text=prompt, images=image, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(model.device)
    generate_ids = model.generate(**inputs, max_new_tokens=50)
    caption = model_processor.batch_decode(generate_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    append_result_to_file({"image_id": image_id, "caption": caption}, file_path)














# val


dataset = load_dataset("lmms-lab/COCO-Caption", cache_dir=DATASETS_CACHE_DIR)
val = dataset['val']

file_path = '/home/hpc/b207dd/b207dd11/llava/captions_val.json'


for item in tqdm(val):
    image_id = int(item['question_id'].split('_')[-1].split('.')[0])  
    caption = item['answer'][0]  
    append_result_to_file({"image_id": image_id, "caption": caption}, file_path)

