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




llava_name = "llava-hf/llava-1.5-7b-hf"
bakllava_name = "llava-hf/bakLlava-v1-hf"
mistral_name = "mistralai/Mistral-7B-v0.1"
mixtral_name = "mistralai/Mixtral-8x7B-v0.1"


TRANSFORMERS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/cache/huggingface/transformers"
DATASETS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/cache/huggingface/datasets"
WORK_DIR = '/home/atuin/b207dd/b207dd11/'


dataset = load_dataset("lmms-lab/COCO-Caption", cache_dir=DATASETS_CACHE_DIR, split='val')
val = dataset.select_columns(['question_id', 'image', 'question', 'answer', 'id', 'file_name', 'coco_url'])

dataset = load_dataset("yerevann/coco-karpathy", cache_dir=DATASETS_CACHE_DIR, split='test')
test = dataset.select_columns(['filepath', 'sentids', 'filename', 'imgid', 'sentences', 'cocoid', 'url'])


karpathy_test_file_names = []
for row in test:
    karpathy_test_file_names.append(row['filename'])

# coco-karpathy test set is a subset of original coco dataset
# instead of downloading images again, here we filter rows of karpathy-test out 
karpathy_rows = []
for row in tqdm(val):
    if row['file_name'] in karpathy_test_file_names:
        image_id = int(row['file_name'].split('_')[-1].split('.')[0])
        image = row['image']
        question = row['question']
        answer = row['answer']
        karpathy_rows.append((image_id, image, question, answer))

with open(WORK_DIR+'cache.pkl', 'wb') as f:
    pickle.dump(karpathy_rows, f)


with open(WORK_DIR+'cache.pkl', 'rb') as f:
    karpathy_rows = pickle.load(f)


### get our model's results on karpathy test set


pretrained_model_dir = '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-pretrained-2/'

model_processor = AutoProcessor.from_pretrained(
    pretrained_model_dir
)


model = LlavaForConditionalGeneration.from_pretrained(
    pretrained_model_dir,
    device_map="auto",
    torch_dtype="auto"
)

model.eval()

#print(dataset)

def append_result_to_file(result, file_path):
    """Appends a single result to a JSON file."""
    with open(file_path, 'a') as file:
        json.dump(result, file)
        file.write(',\n')  # Add a newline to separate JSON objects

result_file_path = '/home/hpc/b207dd/b207dd11/llava/captions_karpathy_val2014_llava-mixtral-pretrained-2_results_2.json'
open(result_file_path, 'w').close()



image_ids = [] 

# write our model's result on karpathy test set to json file for further evaluation
for row in tqdm(karpathy_rows[len(karpathy_rows)//2:]):
    image_id, image, question, answers = row 
    image_ids.append(image_id)
    prompt = question + "<image>"
    inputs = model_processor(text=prompt, images=image, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(model.device)
    generate_ids = model.generate(**inputs, max_new_tokens=64)
    caption = model_processor.batch_decode(generate_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    append_result_to_file({"image_id": image_id, "caption": caption}, result_file_path)


# followed by \coco-caption\coco_karpathy_eval.py














# val


dataset = load_dataset("lmms-lab/COCO-Caption", cache_dir=DATASETS_CACHE_DIR)
val = dataset['val']

file_path = '/home/hpc/b207dd/b207dd11/llava/captions_val.json'


for item in tqdm(val):
    image_id = int(item['question_id'].split('_')[-1].split('.')[0])  
    caption = item['answer'][0]  
    append_result_to_file({"image_id": image_id, "caption": caption}, file_path)





