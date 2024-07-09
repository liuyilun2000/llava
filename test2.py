
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


TRANSFORMERS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/cache/huggingface/transformers"

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B", cache_dir=TRANSFORMERS_CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B", cache_dir=TRANSFORMERS_CACHE_DIR)

DATASETS_CACHE_DIR = "/home/vault/b207dd/b207dd11/cache/huggingface/datasets"
WORK_DIR = '/home/atuin/b207dd/b207dd11/'

llava_name = "llava-hf/llava-1.5-7b-hf"
bakllava_name = "llava-hf/bakLlava-v1-hf"
mistral_name = "mistralai/Mistral-7B-v0.1"
mixtral_name = "mistralai/Mixtral-8x7B-v0.1"



bakllava_processor = AutoProcessor.from_pretrained(
    bakllava_name, 
    cache_dir=TRANSFORMERS_CACHE_DIR
)

'''
#llava_model = LlavaForConditionalGeneration.from_pretrained(llava_name, cache_dir=TRANSFORMERS_CACHE_DIR)
bakllava_model = LlavaForConditionalGeneration.from_pretrained(
    bakllava_name, 
    torch_dtype="auto",#torch.bfloat16,
    device_map="auto",
    cache_dir=TRANSFORMERS_CACHE_DIR
)
#mistral_model = AutoModelForCausalLM.from_pretrained(
#    mistral_name, 
#    #torch_dtype="auto",
#    cache_dir=TRANSFORMERS_CACHE_DIR
#)
mixtral_model = AutoModelForCausalLM.from_pretrained(
    mixtral_name, 
    torch_dtype=torch.bfloat16,
    #torch_dtype="auto",
    cache_dir=TRANSFORMERS_CACHE_DIR
)#, device_map="auto")

mixtral_tokenizer = AutoTokenizer.from_pretrained(mixtral_name, cache_dir=TRANSFORMERS_CACHE_DIR)
#mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_name, cache_dir=TRANSFORMERS_CACHE_DIR)



new_model = bakllava_model


# Abandoned: modifying multi-modal part
while False:
    E_bakllava_original = copy.deepcopy(bakllava_model.language_model.model.embed_tokens.weight.detach())
    E_bakllava = copy.deepcopy(E_bakllava_original[:32000])
    E_mixtral = copy.deepcopy(mixtral_model.model.embed_tokens.weight.detach())
    #E_mistral = copy.deepcopy(mistral_model.model.embed_tokens.weight.detach())


    E_bakllava = E_bakllava.to('cuda')
    E_mixtral = E_mixtral.to('cuda')
    #E_mistral = E_mistral.to('cuda')
    U, S, V = torch.svd(E_bakllava)

    # Invert S with a threshold to avoid division by very small numbers
    threshold = 1e-10
    S_inv = torch.diag(1.0 / S[S > threshold])

    E_bakllava_pinv = V[:, :S_inv.size(0)].matmul(S_inv).matmul(U.t()[:S_inv.size(0), :])

    T = E_bakllava_pinv @ E_mixtral
    #T = E_bakllava_pinv @ E_mistral

    T = T.to(bakllava_model.device)
    transform_layer = torch.nn.Linear(T.shape[0], T.shape[1])
    with torch.no_grad():
        transform_layer.weight.copy_(T)
        transform_layer.bias.fill_(0)

    class TransformedLlavaMultiModalProjector(torch.nn.Module):
        def __init__(self, original_projector, transform_layer):
            super().__init__()
            self.original_projector = original_projector
            self.transform_layer = transform_layer
        def forward(self, image_features):
            hidden_states = self.original_projector(image_features)
            hidden_states = self.transform_layer(hidden_states)
            return hidden_states

    original_projector = new_model.multi_modal_projector
    new_model.multi_modal_projector = TransformedLlavaMultiModalProjector(original_projector, transform_layer)


# adapt LLM to llava setting
new_tokenizer = copy.copy(mixtral_tokenizer)
new_language_model = copy.copy(mixtral_model)

new_tokenizer.add_tokens(['<image>','<pad>'])
new_language_model.resize_token_embeddings(len(new_tokenizer))



new_model.language_model = new_language_model#.to(new_model.device)#mixtral_model
new_model.config.text_config = new_language_model.config

new_model.save_pretrained('/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-test01/')
'''

# '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-test01/model-00001-of-00039.safetensors'


from safetensors import safe_open

tensors = {}
for i in range(39):
    print('#####', i+1)
    with safe_open(f"/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-test01/model-{i+1:05d}-of-00039.safetensors", framework="pt", device=0) as f:
        print(f.keys())





new_model = LlavaForConditionalGeneration.from_pretrained(
    '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-test01/',
    device_map='auto',
    torch_dtype="auto"
)
    #, ignore_mismatched_sizes=True)

'''
upload_folder(
    folder_path='/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-test01',
    repo_id='liuyilun2000/llava-mixtral-test01',
    repo_type='model',
    multi_commits=True,
    multi_commits_verbose=True
)
new_model.push_to_hub("liuyilun2000/llava-mixtral-test01",token='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')
'''

#model = bakllava_model
model = new_model

for param in model.vision_tower.parameters():
    param.requires_grad = False


for param in model.language_model.parameters():
    param.requires_grad = False


for param in model.multi_modal_projector.parameters():
    param.requires_grad = True

#model.language_model.config.torch_dtype="float32"




'''
def convert_image_dataset(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        data = json.load(infile)
        for entry in data:
            image_path = entry['image']
            #if '00000' <= image_path.split('/')[0] <= '00010':
            text_input = next((conv['value'] for conv in entry['conversations'] if conv['from'] == 'human'), None)
            text_output = next((conv['value'] for conv in entry['conversations'] if conv['from'] == 'gpt'), None)
            new_entry = {
                "file_name": image_path,
                "text_input": text_input,
                "text_output": text_output
            }
            json.dump(new_entry, outfile)
            outfile.write('\n')

input_file = WORK_DIR+"LLaVA-Pretrain/blip_laion_cc_sbu_558k.json.bak"
output_file = WORK_DIR+"LLaVA-Pretrain/metadata.jsonl"
convert_image_dataset(input_file, output_file)
'''



data_dir = WORK_DIR+"LLaVA-Pretrain"
dataset = load_dataset("imagefolder", data_dir=data_dir, cache_dir=DATASETS_CACHE_DIR)

dataset_train = dataset['train'].train_test_split(test_size=548128/558128)
#558128/16=34883
split=[f"train[{k}%:{k+10}%]" for k in range(0, 100, 10)])
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
    batched=True, batch_size=128, 
    remove_columns=['image', 'text_input', 'text_output']
)
processed_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'pixel_values'])
# DataLoader

'''
with open(WORK_DIR+'processed_dataset.pkl', 'wb') as f:
    pickle.dump(processed_dataset, f)
''' 

with open(WORK_DIR+'processed_dataset.pkl', 'rb') as f:
    processed_dataset = pickle.load(f)


device = model.device
processed_dataset = processed_dataset.with_format("torch", device=device)
train_loader = DataLoader(processed_dataset, batch_size=10, shuffle=False)

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

num_epochs=1


#from torch.cuda.amp import autocast


#model=bakllava_model
#with autocast(dtype=torch.bfloat16):
model.train()
for epoch in range(num_epochs):
    for batch_input in tqdm(train_loader):
        outputs = model(**batch_input)
        loss = outputs.loss
        #break
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()


input_ids=batch_input['input_ids']#.to(torch.Int)
attention_mask=batch_input['attention_mask']
pixel_values=batch_input['pixel_values'].to(torch.bfloat16)
'''
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(bakllava_processor.tokenizer, mlm=False)


training_args = TrainingArguments(
    output_dir='/home/vault/b207dd/b207dd11/llava-mixtral',          # output directory
    num_train_epochs=1,              # total number of training epochs
    #per_device_train_batch_size=16,  # batch size per device during training
    #warmup_steps=500,                # number of warmup steps for learning rate scheduler
    #weight_decay=0.01,               # strength of weight decay
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
)

trainer.train()
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









config = LoraConfig(
    r=16,
    lora_alpha=16,
    #target_modules=["block_sparse_moe"],
    #target_modules=["self_attn"],
    target_modules=["q_proj", "k_proj"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
lora_model = get_peft_model(model, config)
merged_model = lora_model.merge_and_unload()





def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


for param in new_model.vision_tower.parameters():
    param.requires_grad = False

for param in new_model.language_model.parameters():
    param.requires_grad = False










TRANSFORMERS_CACHE_DIR = "/home/vault/b207dd/b207dd11/cache/huggingface/transformers"
DATASETS_CACHE_DIR = "/home/vault/b207dd/b207dd11/cache/huggingface/datasets"

dataset = load_dataset("liuhaotian/LLaVA-CC3M-Pretrain-595K", cache_dir=DATASETS_CACHE_DIR)

dataset = load_dataset("liuhaotian/LLaVA-Pretrain", cache_dir=DATASETS_CACHE_DIR)

dataset = load_dataset("SALT-NLP/LLaVAR", cache_dir=DATASETS_CACHE_DIR)


# Benchmark
dataset = load_dataset("Andyrasika/VQA-Dataset", cache_dir=DATASETS_CACHE_DIR)
dataset = load_dataset("HuggingFaceM4/VQAv2", cache_dir=DATASETS_CACHE_DIR)


#print_trainable_parameters(new_model)
print_trainable_parameters(lora_model)