import os

from PIL import Image
import copy
import numpy as np
import requests
import json

import pickle
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from datasets import load_dataset
#from huggingface_hub import upload_file, upload_folder

import transformers
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import Trainer, TrainingArguments
from transformers import AdamW
from transformers import get_scheduler
'''
import accelerate
from accelerate import Accelerator
'''

import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec

from deepspeed import comm as dist
deepspeed.init_distributed()
rank = dist.get_rank()
print('Deepspeed initiallized!', rank, '/', dist.get_world_size())

import peft
from peft import LoraConfig, get_peft_model

print(f"Transformers version: {transformers.__version__}")
#print(f"Accelerate version: {accelerate.__version__}")
print(f"PEFT version: {peft.__version__}")


'''
accelerator = Accelerator()
device = accelerator.device
print(device, device==torch.device(0))
print(accelerator.num_processes, accelerator.process_index)
'''

TRANSFORMERS_CACHE_DIR = "/home/vault/b207dd/b207dd11/cache/huggingface/transformers"
DATASETS_CACHE_DIR = "/home/vault/b207dd/b207dd11/cache/huggingface/datasets"
WORK_DIR = '/home/atuin/b207dd/b207dd11/'

llava_name = "llava-hf/llava-1.5-7b-hf"
bakllava_name = "llava-hf/bakLlava-v1-hf"
mistral_name = "mistralai/Mistral-7B-v0.1"
mixtral_name = "mistralai/Mixtral-8x7B-v0.1"

if rank==0:
    bakllava_processor = AutoProcessor.from_pretrained(
        bakllava_name, 
        cache_dir=TRANSFORMERS_CACHE_DIR
    )


'''
with init_empty_weights():
    model = LlavaForConditionalGeneration()

model = load_checkpoint_and_dispatch(
    model, 
    checkpoint='/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-test01/', 
    device_map="auto"
)

model = LlavaForConditionalGeneration.from_pretrained(
    '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-test01/',
    device_map=accelerator.device,
    torch_dtype="auto"
)
'''

#device = torch.device(0)
'''
if device == torch.device(0):
    device0 = 0
    device1 = 1
    device2 = 2
    device3 = 3    
else:
    device0 = 4
    device1 = 5
    device2 = 6
    device3 = 7

device_map={
    "vision_tower": device0, "multi_modal_projector": device0, 
    "language_model.model.embed_tokens": device0,
    "language_model.model.norm": device3,
    "language_model.lm_head": device3
}
for layer in range(32):
    if layer < 8:
        device_map["language_model.model.layers."+str(layer)] = device0
    elif layer < 16:
        device_map["language_model.model.layers."+str(layer)] = device1
    elif layer < 24:
        device_map["language_model.model.layers."+str(layer)] = device2
    else: 
        device_map["language_model.model.layers."+str(layer)] = device3
'''


device0 = 0
device1 = 1
device2 = 2
device3 = 3
device4 = 4
device5 = 5
device6 = 6
device7 = 7

device_map={
    "vision_tower": device0, "multi_modal_projector": device0, 
    "language_model.model.embed_tokens": device0,
    "language_model.model.norm": device7,
    "language_model.lm_head": device7
}

for layer in range(32):
    if layer <= 0:
        device_map["language_model.model.layers."+str(layer)] = device0
    elif layer <= 5:
        device_map["language_model.model.layers."+str(layer)] = device1
    elif layer <= 10:
        device_map["language_model.model.layers."+str(layer)] = device2
    elif layer <= 15:
        device_map["language_model.model.layers."+str(layer)] = device3
    elif layer <= 20:
        device_map["language_model.model.layers."+str(layer)] = device4
    elif layer <= 25:
        device_map["language_model.model.layers."+str(layer)] = device5
    elif layer <= 30:
        device_map["language_model.model.layers."+str(layer)] = device6
    else: 
        device_map["language_model.model.layers."+str(layer)] = device7

model = LlavaForConditionalGeneration.from_pretrained(
    '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-test01/',
    #device_map="sequential",
    device_map=device_map,
    torch_dtype="auto",
    #max_memory = {0: "36GIB", 1: "36GIB", 2: "36GIB", 3: "36GIB", 4: "36GIB", 5: "36GIB", 6: "36GIB", 7: "36GIB"}
)




for param in model.vision_tower.parameters():
    param.requires_grad = False

for param in model.language_model.parameters():
    param.requires_grad = False

for param in model.multi_modal_projector.parameters():
    param.requires_grad = True


from torch.nn import Sequential

pipeline_stages = Sequential(
    model.vision_tower,
    model.language_model.model.embed_tokens,
    model.multi_modal_projector,
    Sequential(*model.language_model.model.layers[:8]),
    Sequential(*model.language_model.model.layers[8:16]),
    Sequential(*model.language_model.model.layers[16:24]),
    Sequential(*model.language_model.model.layers[24:]),
    model.language_model.model.norm,
    model.language_model.lm_head
)



from deepspeed.pipe import PipelineModule
net = PipelineModule(layers=pipeline_stages, num_stages=8)

print(net)

'''
data_dir = WORK_DIR+"LLaVA-Pretrain"
dataset = load_dataset("imagefolder", data_dir=data_dir, cache_dir=DATASETS_CACHE_DIR)

dataset_train = dataset['train'].train_test_split(test_size=548128/558128)
dataset_train = dataset_train['train']

def preprocess_function(examples):
    images = examples['image']
    inputs = bakllava_processor(text=[a+b for a,b in zip(examples['text_input'],examples['text_output'])], images=images, return_tensors="pt", padding=True)
    #labels = bakllava_processor.tokenizer(examples['text_output'], return_tensors="pt", padding=True, truncation=True)["input_ids"]
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"], 
        "pixel_values": inputs["pixel_values"]
    }

processed_dataset = dataset_train.map(
    preprocess_function, 
    batched=True, batch_size=10, 
    remove_columns=['image', 'text_input', 'text_output']
)
processed_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'pixel_values'])
# DataLoader

with open(WORK_DIR+'processed_dataset.pkl', 'wb') as f:
    pickle.dump(processed_dataset, f)
''' 

with open(WORK_DIR+'processed_dataset.pkl', 'rb') as f:
    processed_dataset = pickle.load(f)

device = model.device
processed_dataset = processed_dataset.with_format("torch")#, device=device)
train_dataloader = DataLoader(processed_dataset, batch_size=10, shuffle=False)

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)


num_epochs = 1
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)


model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)


num_epochs=1

model.train()
for epoch in range(num_epochs):
    for batch_input in tqdm(train_dataloader):
        optimizer.zero_grad()
        outputs = model(**batch_input)        
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        torch.cuda.empty_cache()

accelerator.wait_for_everyone()

unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(WORK_DIR, safe_serialization=True)


prompt = "<pad><image>\nUSER: What's the content of the image?\nASSISTANT:"
image_path = "/home/hpc/b207dd/b207dd11/llava/australia.jpg"
image = Image.open(image_path)

device = model.device
inputs = bakllava_processor(text=prompt, images=image, return_tensors="pt")
for k,v in inputs.items():
    v = v.to(device)

input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
pixel_values = inputs['pixel_values']
labels = input_ids.detach().clone()
'''
output = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    pixel_values=pixel_values,
    labels=labels
)
output.loss
'''
import time
start_time = time.time()
output_pipeline = model_pipeline((input_ids, pixel_values, attention_mask, labels))
output_pipeline[0]
print("--- %s seconds ---" % (time.time() - start_time))

import time
start_time = time.time()

generate_ids = model.generate(**inputs, max_new_tokens=20)

print("--- %s seconds ---" % (time.time() - start_time))
bakllava_processor.batch_decode(generate_ids)#, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]



'''
    for batch_input in tqdm(train_dataloader):
        outputs = model(**batch_input)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
'''



'''
#prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
prompt = '<image>Describe the image concisely.'
prompt = '<image>Render a clear and concise summary of the photo.'
prompt = '<image>What is this?'

prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
image_path = "/home/hpc/b207dd/b207dd11/llava/australia.jpg"
image = Image.open(image_path)

device = model.device
inputs = bakllava_processor(text=prompt, images=image, return_tensors="pt")
for k,v in inputs.items():
    v = v.to(device)

generate_ids = model.generate(**inputs, max_new_tokens=20)
bakllava_processor.batch_decode(generate_ids)#, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


'''