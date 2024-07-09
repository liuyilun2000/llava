import random
random.seed(42)

from PIL import Image
import requests
import json
import torch
import os
import shutil
import transformers
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import Trainer, TrainingArguments
from transformers import AdamW

from torch.utils.data import DataLoader


from datasets import load_dataset, Dataset, Features, Value, Image
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
MAP_BATCH_SIZE = 128

llava_name = "llava-hf/llava-1.5-7b-hf"
bakllava_name = "llava-hf/bakLlava-v1-hf"
mistral_name = "mistralai/Mistral-7B-v0.1"
mixtral_name = "mistralai/Mixtral-8x7B-v0.1"


pretrained_model_dir = '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-pretrained-2/'

model_processor = AutoProcessor.from_pretrained(pretrained_model_dir)




'''
data_dir = WORK_DIR+"LLaVA-Instruct-150K/"

def separate_json(file_path, output_path_with_images, output_path_without_images):
    with open(file_path, 'r') as f:
        data = json.load(f)    
    data_with_images = []
    data_without_images = []
    for item in data:
        if "image" in item:
            data_with_images.append(item)
        else:
            data_without_images.append(item)    
    with open(output_path_with_images, 'w') as f:
        json.dump(data_with_images, f, indent=4)    
    with open(output_path_without_images, 'w') as f:
        json.dump(data_without_images, f, indent=4)

input_file = data_dir+'llava_v1_5_mix665k.json'
output_file_with_images = data_dir+'llava_v1_5_mix665k_with_images.json'
output_file_without_images = data_dir+'llava_v1_5_mix665k_without_images.json'

separate_json(input_file, output_file_with_images, output_file_without_images)
'''

'''
error_path_list = [] 
def validate_json_structure(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    all_entries_valid = True
    for entry in tqdm(data):
        # Check if 'image' key exists and is a valid path
        if 'image' not in entry:
            print(f"Entry {entry['id']} is missing 'image' key.")
            all_entries_valid = False
        else:
            tmp_path = '/home/atuin/b207dd/b207dd11/LLaVA-mix665k-with-images/' + entry['image']
            if not os.path.exists(tmp_path):
                error_path_list.append(tmp_path)
                print(f"Image path {tmp_path} for entry {entry['id']} does not exist.")
                all_entries_valid = False
        # Check if 'conversations' key exists and is a list
        if 'conversations' not in entry:
            print(f"Entry {entry['id']} is missing 'conversations' key.")
            all_entries_valid = False
        elif not isinstance(entry['conversations'], list):
            print(f"'conversations' for entry {entry['id']} is not a list.")
            all_entries_valid = False
        else:
            # Check if each conversation is valid
            for conversation in entry['conversations']:
                if 'from' not in conversation or 'value' not in conversation:
                    print(f"Invalid conversation structure in entry {entry['id']}.")
                    all_entries_valid = False
    if all_entries_valid:
        print("All entries are valid.")

validate_json_structure('/home/atuin/b207dd/b207dd11/LLaVA-mix665k-with-images/metadata.json')


with open('/home/atuin/b207dd/b207dd11/LLaVA-mix665k-with-images/ocr_vqa/dataset.json', 'r') as fp:
    data = json.load(fp)

for error_path in error_path_list:
    if not os.path.exists(error_path):
        # Extract the image ID from the error path
        image_id = os.path.splitext(os.path.basename(error_path))[0]
        if image_id in data:
            ext=os.path.splitext(data[image_id]['imageURL'])[1]
            outputFile='images/%s%s'%(image_id,ext)
            #pdb.set_trace()
            if not os.path.exists(outputFile):
                print('not downloaded', image_url, outputFile)
            else:
                print('downloaded', image_url, outputFile)
            #ureq.urlretrieve(data[k]['imageURL'],outputFile)  
            #redownload_image(image_url, error_path)
        else:
            print(f"Image ID {image_id} not found in the dataset.")

source_image_path = '/home/atuin/b207dd/b207dd11/LLaVA-mix665k-with-images/ocr_vqa/images/716KF1XB6KL.jpg'

for error_path in error_path_list:
    try:
        shutil.copy(source_image_path, error_path)
    except Exception as e:
        print(f"Failed to copy to {error_path}: {e}")
'''





'''
def check_conversation_values(data):
    index = 0
    for entry in tqdm(data):
        image_token_counter = 0
        for conversation in entry.get('conversations', []):
            while '###' in conversation['value']:
                print(index, '### removed into ##')
                conversation['value'] = conversation['value'].replace('###', '##')
            if conversation['value'].startswith('#'):
                print(index, '# at bos added \' \'')
                conversation['value'] = ' ' + conversation['value']
            if conversation['value'].endswith('#'):
                print(index, '# at eos added .\\n')
                conversation['value'] += '.\n'

def check_conversation_froms(data):
    index = 0
    froms = []
    for entry in tqdm(data):
        image_token_counter = 0
        for conversation in entry.get('conversations', []):
            if conversation['from'] in froms:
                continue
            else:
                froms.append(conversation['from'])
    return froms

def check_conversation_image_count(data):
    index = 0
    froms = []
    for entry in tqdm(data):
        image_token_counter = 0
        for conversation in entry.get('conversations', []):
            if '<image>' in conversation['value']:
                image_token_counter += 1
        if image_token_counter != 1:
            print(index, entry)
        index += 1
    return froms

def check_conversation_image_position(data):
    index = 0
    image_start_count = 0
    image_end_count = 0
    image_mid_count = 0
    for entry in tqdm(data):
        image_token_counter = 0
        for conversation in entry.get('conversations', []):
            if conversation['from'] == 'human':
                if conversation['value'].startswith('<image>'):
                    image_start_count += 1
                    break
                elif conversation['value'].endswith('<image>'):
                    image_end_count += 1
                    break
                else:
                    image_mid_count += 1
                    break
    return image_start_count, image_end_count, image_mid_count

stop_token = '###'
def combine_conversation_values_with_images(data):
    index = 0
    for entry in tqdm(data):
        text = ''
        for conversation in entry.get('conversations', []):
            if conversation['from'] == 'human':
                text += 'Human: ' + conversation['value'] + stop_token
            else:
                text += 'Assistant: <s>' + conversation['value'] + '</s>' + stop_token
        entry['file_name'] = entry['image']
        entry['text'] = text
        del entry['id']
        del entry['image']
        del entry['conversations']

def combine_conversation_values_plain_texts(data):
    index = 0
    for entry in tqdm(data):
        text = ''
        for conversation in entry.get('conversations', []):
            if conversation['from'] == 'human':
                text += 'Human: ' + conversation['value'] + stop_token
            else:
                text += 'Assistant: <s>' + conversation['value'] + '</s>' + stop_token
        entry['text'] = text
        del entry['id']
        del entry['model']
        del entry['conversations']

with open('/home/atuin/b207dd/b207dd11/LLaVA-mix665k-with-images/metadata.json.bak', 'r') as file:
    data_with_images = json.load(file)

check_conversation_values(data_with_images)
check_conversation_froms(data_with_images)
check_conversation_image_count(data_with_images)
check_conversation_image_position(data_with_images)
combine_conversation_values_with_images(data_with_images)

with open('/home/atuin/b207dd/b207dd11/LLaVA-mix665k-with-images/metadata.jsonl', 'w', encoding='utf-8') as file:
    for entry in tqdm(data_with_images):
        json.dump(entry, file)
        file.write('\n')


with open('/home/atuin/b207dd/b207dd11/LLaVA-mix665k-plain-texts/metadata.json', 'r', encoding='utf-8') as file:
    data_plain_texts = json.load(file)

check_conversation_values(data_plain_texts)
check_conversation_froms(data_plain_texts)
combine_conversation_values_plain_texts(data_plain_texts)

with open('/home/atuin/b207dd/b207dd11/LLaVA-mix665k-plain-texts/metadata.jsonl', 'w', encoding='utf-8') as file:
    for entry in tqdm(data_plain_texts):
        json.dump(entry, file)
        file.write('\n')
'''








with open('/home/atuin/b207dd/b207dd11/LLaVA-mix665k-with-images/metadata.jsonl', 'r', encoding='utf-8') as file:
    data_with_images_json_list = list(file)

with open('/home/atuin/b207dd/b207dd11/LLaVA-mix665k-plain-texts/metadata.jsonl', 'r', encoding='utf-8') as file:
    data_plain_texts_json_list = list(file)



data_with_images_base_path = '/home/atuin/b207dd/b207dd11/LLaVA-mix665k-with-images/'
data_with_images_dict = {
    'image': [],
    'text': []
}
for json_str in data_with_images_json_list:
    entry = json.loads(json_str)
    image_path = os.path.join(data_with_images_base_path, entry['file_name'])
    data_with_images_dict['image'].append(image_path)
    data_with_images_dict['text'].append(entry['text'])

data_with_images_features = Features({
    'image': Image(), 
    'text': Value('string'),
})
data_with_images_dataset = Dataset.from_dict(data_with_images_dict, features=data_with_images_features)


data_plain_texts_dict = {'text': []}
for json_str in data_plain_texts_json_list:
    entry = json.loads(json_str)
    data_plain_texts_dict['text'].append(entry['text'])

data_plain_texts_dataset = Dataset.from_dict(data_plain_texts_dict)




print(len(data_with_images_dataset), len(data_plain_texts_dataset))


system_message = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
stop_token = '###'
bos_token_id = model_processor.tokenizer.bos_token_id
eos_token_id = model_processor.tokenizer.eos_token_id
pad_token_id = model_processor.tokenizer.pad_token_id


def preprocess_function(examples):
    if 'image' in examples.keys():
        images = examples['image']
    else:
        images = None
    # the processor automatically adds '<s>' at the start of the sequence
    # we've already inserted '<s>' and '</s>' manually at the both end of labels we want to train model with to help locate the boundary
    texts = []
    for t in examples['text']:
        texts.append(system_message + stop_token + t)
    #print(texts)
    inputs = model_processor(text=texts, images=images, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    new_input_ids = []
    new_attention_mask = []
    labels = []
    max_length = 0
    for i, row in enumerate(input_ids):
        # Find all indices of '<s>' and '</s>' tokens
        sep_indices = (row == bos_token_id).nonzero(as_tuple=True)[0]
        eos_indices = (row == eos_token_id).nonzero(as_tuple=True)[0]
        if len(sep_indices) >= 2 and len(eos_indices) == len(sep_indices) - 1 :
            # Initialize label tensor with -100 (mask value)
            label_row = torch.full_like(row, -100)
            # Iterate over the pairs of <s> and </s> tokens to keep the text between them
            for start_idx, end_idx in zip(sep_indices[1:], eos_indices+1):
                label_row[start_idx + 1:end_idx] = row[start_idx + 1:end_idx]
            # Remove the second '<s>' and '</s>' tokens from input_ids and attention_mask
            mask = torch.ones_like(row, dtype=torch.bool)
            mask[sep_indices[1:]] = False
            #mask[eos_indices] = False
            new_input_ids_row = row[mask]
            new_attention_mask_row = attention_mask[i][mask]
            new_label_row = label_row[mask]
            if len(new_input_ids_row) == len(new_attention_mask_row) == len(new_label_row):
                if max_length < len(new_input_ids_row):
                    max_length = len(new_input_ids_row)
                new_input_ids.append(new_input_ids_row)
                new_attention_mask.append(new_attention_mask_row)
                labels.append(new_label_row)
            else:
                print(f"Length Error: unexpected input structure in {examples}")
                continue
        else:
            print(f"Label Boundary Error: unexpected input structure in {examples}")
            new_input_ids.append(input_ids[i])
            new_attention_mask.append(attention_mask[i])
            labels.append(input_ids[i])
            continue
    #stack with pad
    new_input_ids = torch.nn.utils.rnn.pad_sequence(new_input_ids, batch_first=True, padding_value=pad_token_id)
    new_attention_mask = torch.nn.utils.rnn.pad_sequence(new_attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    #print(labels, new_input_ids, new_attention_mask)
    if 'image' in examples.keys():
        return {
            "input_ids": new_input_ids,
            "attention_mask": new_attention_mask,
            "pixel_values": pixel_values,
            "labels": labels
        }
    else:
        return {
            "input_ids": new_input_ids,
            "attention_mask": new_attention_mask,
            "labels": labels
        }




dataset = data_with_images_dataset
DATASET_SIZE = len(dataset) # len()

indices = list(range(DATASET_SIZE))
random.shuffle(indices)

skip_shard_id = 0
shard_id = 0
for shard_start in tqdm(range(0, DATASET_SIZE, MAP_BATCH_SIZE)):
    if shard_id < skip_shard_id:
        shard_id += 1
        continue
    shard_end = min(DATASET_SIZE, shard_start+MAP_BATCH_SIZE)
    print("Selecting range:", shard_start, '-', shard_end, "for shard", shard_id)
    dataset_train = dataset.select(indices[shard_start:shard_end])
    processed_dataset = dataset_train.map(
        preprocess_function,
        batched=True, batch_size=len(dataset_train), 
        remove_columns=['image', 'text']#, num_proc=8
    )
    processed_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'pixel_values', 'labels'])    
    processed_dataset.save_to_disk(WORK_DIR+'LLaVA-PEFT_processed_dataset_with_images/shard_'+str(shard_id))
    shard_id += 1





dataset = data_plain_texts_dataset
DATASET_SIZE = len(dataset) 

indices = list(range(DATASET_SIZE))
random.shuffle(indices)

skip_shard_id = 0
shard_id = 0
for shard_start in tqdm(range(0, DATASET_SIZE, MAP_BATCH_SIZE)):
    if shard_id < skip_shard_id:
        shard_id += 1
        continue
    shard_end = min(DATASET_SIZE, shard_start+MAP_BATCH_SIZE)
    print("Selecting range:", shard_start, '-', shard_end, "for shard", shard_id)
    dataset_train = dataset.select(indices[shard_start:shard_end])
    processed_dataset = dataset_train.map(
        preprocess_function,
        batched=True, batch_size=len(dataset_train), 
        remove_columns=['text']#, num_proc=8
    )
    processed_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])    
    processed_dataset.save_to_disk(WORK_DIR+'LLaVA-PEFT_processed_dataset_plain_texts/shard_'+str(shard_id))
    shard_id += 1





NUM_IMAGE_SHARDS = 488
NUM_TEXT_SHARDS = 32
NUM_DATASET_SHARDS = NUM_IMAGE_SHARDS + NUM_TEXT_SHARDS

image_shard_dir = 'image_'#'/home/atuin/b207dd/b207dd11/LLaVA-PEFT_processed_dataset_with_images'
text_shard_dir = 'text_'#'/home/atuin/b207dd/b207dd11/LLaVA-PEFT_processed_dataset_plain_texts'
image_shards = [os.path.join(image_shard_dir, f'shard_{i}') for i in range(NUM_IMAGE_SHARDS)]
text_shards = [os.path.join(text_shard_dir, f'shard_{i}') for i in range(NUM_TEXT_SHARDS)]
random.seed(42)
random.shuffle(image_shards)
random.shuffle(text_shards)

interval = NUM_IMAGE_SHARDS // NUM_TEXT_SHARDS

dataset_shards = []
text_index = 0
for i in range(NUM_IMAGE_SHARDS):
    dataset_shards.append(image_shards[i])
    if (i + 1) % interval == 0 and text_index < NUM_TEXT_SHARDS:
        dataset_shards.append(text_shards[text_index])
        text_index += 1

dataset_shards