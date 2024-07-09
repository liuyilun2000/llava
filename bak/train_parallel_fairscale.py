import os

import time
from PIL import Image
import copy
import numpy as np
import requests
import json

import pickle
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import Sequential
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from datasets import load_dataset
#from huggingface_hub import upload_file, upload_folder

import transformers
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa

from transformers import Trainer, TrainingArguments
from transformers import AdamW
from transformers import get_scheduler


import fairscale



print(f"Transformers version: {transformers.__version__}")


TRANSFORMERS_CACHE_DIR = "/home/vault/b207dd/b207dd11/cache/huggingface/transformers"
DATASETS_CACHE_DIR = "/home/vault/b207dd/b207dd11/cache/huggingface/datasets"
WORK_DIR = '/home/atuin/b207dd/b207dd11/'

llava_name = "llava-hf/llava-1.5-7b-hf"
bakllava_name = "llava-hf/bakLlava-v1-hf"
mistral_name = "mistralai/Mistral-7B-v0.1"
mixtral_name = "mistralai/Mixtral-8x7B-v0.1"

#if rank==0:
bakllava_processor = AutoProcessor.from_pretrained(
    bakllava_name, 
    cache_dir=TRANSFORMERS_CACHE_DIR
)



class LlavaMultiModalModuleWrapper(nn.Module):
    def __init__(self, config, vision_tower, embed_tokens, multi_modal_projector):
        super().__init__()
        self.config = config
        self.vision_tower = vision_tower
        self.embed_tokens = embed_tokens
        self.multi_modal_projector = multi_modal_projector
        self.device = self.vision_tower.device
    #
    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.config.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)
        #
        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]
        #
        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)
        #
        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]
        #
        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)
        #
        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )
        #
        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)
        #
        if labels is None:
            final_labels = None
        #
        return final_embedding, final_attention_mask, final_labels, position_ids
    #
    def forward(self, inputs):
        self.device = self.vision_tower.device
        input_ids, pixel_values, attention_mask, labels = inputs
        input_ids, pixel_values, attention_mask, labels = input_ids.to(self.device), pixel_values.to(self.device), attention_mask.to(self.device), labels.to(self.device)
        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)
            image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_outputs.hidden_states[self.config.vision_feature_layer]
        if self.config.vision_feature_select_strategy ==  "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.config.vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(
                f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
            )        
        image_features = self.multi_modal_projector(selected_image_feature)
        inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, attention_mask, labels
        )
        batch_size, seq_length, _ = inputs_embeds.shape
        position_ids = position_ids.view(-1, seq_length).long()
        attention_mask_4d = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            0,
            #sliding_window=self.config.text_config.sliding_window,
        )
        hidden_states = inputs_embeds
        torch.cuda.empty_cache()
        return hidden_states, attention_mask, attention_mask_4d, position_ids, labels

class LanguageModelLayerWrapper(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    def forward(self, inputs):
        # Assuming the first element of the tuple is what we need
        hidden_states, attention_mask, attention_mask_4d, position_ids, labels = inputs
        #check_tensor(hidden_states, name="hidden_states")
        #check_tensor(attention_mask_4d, name="attention_mask_4d")
        #check_tensor(position_ids, name="position_ids")
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states=hidden_states, 
                attention_mask=attention_mask_4d, 
                position_ids=position_ids)
            hidden_states = layer_outputs[0]
        torch.cuda.empty_cache()
        return hidden_states, attention_mask, attention_mask_4d, position_ids, labels


class LanguageModelFinalWrapper(nn.Module):
    def __init__(self, norm, lm_head):
        super().__init__()
        self.norm = norm
        self.lm_head = lm_head
    def forward(self, inputs):
        # Assuming the first element of the tuple is what we need
        hidden_states, attention_mask, attention_mask_4d, position_ids, labels = inputs
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        #
        loss = None
        # Shift so that tokens < n predict n
        if attention_mask is not None:
            shift_attention_mask = attention_mask[..., 1:]
            shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()            
            shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()   
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
        )
        loss = loss.reshape(1)
        torch.cuda.empty_cache()
        return loss #logits

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
'''
model = LlavaForConditionalGeneration.from_pretrained(
    '/home/vault/b207dd/b207dd11/llava-mixtral/llava-mixtral-test01/',
    #device_map="sequential",
    #device_map=device_map,
    device_map="cpu",
    torch_dtype="auto"
)

for param in model.vision_tower.parameters():
    param.requires_grad = False

for param in model.language_model.parameters():
    param.requires_grad = False

for param in model.multi_modal_projector.parameters():
    param.requires_grad = True



pipeline_stages = Sequential(
    LlavaMultiModalModuleWrapper(
        model.config, 
        model.vision_tower,
        model.language_model.model.embed_tokens,
        model.multi_modal_projector
    ),
    LanguageModelLayerWrapper(model.language_model.model.layers[:1]),
    LanguageModelLayerWrapper(model.language_model.model.layers[1:6]),
    LanguageModelLayerWrapper(model.language_model.model.layers[6:11]),
    LanguageModelLayerWrapper(model.language_model.model.layers[11:16]),
    LanguageModelLayerWrapper(model.language_model.model.layers[16:21]),
    LanguageModelLayerWrapper(model.language_model.model.layers[21:26]),
    LanguageModelLayerWrapper(model.language_model.model.layers[26:31]),
    LanguageModelLayerWrapper(model.language_model.model.layers[31:]),
    LanguageModelFinalWrapper(
        model.language_model.model.norm,
        model.language_model.lm_head
    )    
)

torch.cuda.mem_get_info(1)
model_pipeline = fairscale.nn.Pipe(pipeline_stages, balance=[2,1,1,1,1,1,1,2], chunks=1)
torch.cuda.mem_get_info(1)


with open(WORK_DIR+'processed_dataset.pkl', 'rb') as f:
    processed_dataset = pickle.load(f)

device = model.device
processed_dataset = processed_dataset.with_format("torch")#, device=device)
train_dataloader = DataLoader(processed_dataset, batch_size=16, shuffle=False)

optimizer = AdamW(filter(lambda p: p.requires_grad, model_pipeline.parameters()), lr=2e-3)


num_epochs = 1
'''
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
'''
device = model_pipeline[0].vision_tower.device
model_pipeline.train()
for epoch in range(num_epochs):
    for batch_input in tqdm(train_dataloader):
        optimizer.zero_grad()
        input_ids = batch_input['input_ids'].to(device)
        attention_mask = batch_input['attention_mask'].to(device)
        pixel_values = batch_input['pixel_values'].to(device)
        labels = input_ids.detach().clone().to(device)
        start_time = time.time()
        loss = model_pipeline((input_ids, pixel_values, attention_mask, labels))
        #print("--- %s seconds ---" % (time.time() - start_time))
        #print(loss)
        loss.backward()
        optimizer.step()
        #lr_scheduler.step()
        torch.cuda.empty_cache()



'''

        start_time = time.time()
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels
        )
        print("--- %s seconds ---" % (time.time() - start_time))
        print(output.loss)
'''

model(input_ids, pixel_values, attention_mask, labels)



accelerator.wait_for_everyone()

unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(WORK_DIR, safe_serialization=True)



prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
image_path = "/home/hpc/b207dd/b207dd11/llava/australia.jpg"
image = Image.open(image_path)

inputs = bakllava_processor(text=prompt, images=image, return_tensors="pt")
'''
device = model.device
for k,v in inputs.items():
    v = v.to(device)
'''

#generate_ids = model_pipeline.generate(**inputs, max_new_tokens=20)

input_ids = input['input_ids']
attention_mask = input['attention_mask']
pixel_values = input['pixel_values']
labels = input_ids.detach().clone().to(input_ids.device)
print(input_ids.device, attention_mask.device, pixel_values.device, labels.device)
outputs = model_pipeline((input_ids, pixel_values, attention_mask, attention_mask, labels))

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